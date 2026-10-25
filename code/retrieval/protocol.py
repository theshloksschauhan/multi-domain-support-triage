"""JSON shapes shared by shard servers and clients.

The server binds to localhost and caps request bodies. It is a local
multi-process protocol, not a service for untrusted networks.
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Sequence

from config.settings import DEFAULT_BODY_LIMIT
from data_loader.corpus_loader import Document


MAX_TOP_K = 1000


class ProtocolError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def parse_object(raw: bytes, limit: int = DEFAULT_BODY_LIMIT) -> dict:
    if len(raw) > limit:
        raise ProtocolError(413, "request body is too large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(400, "malformed JSON") from exc
    if not isinstance(data, dict):
        raise ProtocolError(400, "JSON body must be an object")
    return data


def document_to_payload(doc: Document) -> dict:
    return {
        "doc_id": doc.doc_id,
        "company": doc.company,
        "product_area": doc.product_area,
        "text": doc.text,
        "source": str(doc.path),
        "tokens": list(doc.tokens),
        "term_freq": {term: int(tf) for term, tf in doc.term_freq.items()},
    }


def document_from_payload(item: object) -> Document:
    if not isinstance(item, dict):
        raise ProtocolError(400, "document must be an object")
    required = ("doc_id", "company", "product_area", "text", "source", "tokens", "term_freq")
    missing = [key for key in required if key not in item]
    if missing:
        raise ProtocolError(400, "document missing " + ", ".join(missing))
    if not isinstance(item["tokens"], list) or not isinstance(item["term_freq"], dict):
        raise ProtocolError(400, "document tokens and term_freq have the wrong shape")
    try:
        frequencies = {str(term): int(value) for term, value in item["term_freq"].items()}
        tokens = [str(token) for token in item["tokens"]]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(400, "document term frequencies must be integers") from exc
    return Document(
        doc_id=str(item["doc_id"]),
        company=str(item["company"]),
        product_area=str(item["product_area"]),
        text=str(item["text"]),
        tokens=tokens,
        term_freq=Counter(frequencies),
        path=Path(str(item["source"])),
    )


def documents_from_batch(payload: dict) -> List[Document]:
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ProtocolError(400, "documents must be a list")
    return [document_from_payload(item) for item in documents]


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(400, f"{label} must be an integer")
    return value


def commit_request(payload: dict) -> tuple:
    generation = payload.get("generation")
    if not isinstance(generation, str) or not generation:
        raise ProtocolError(400, "commit requires a generation string")
    expected = _as_int(payload.get("expected_count"), "expected_count")
    if expected < 0:
        raise ProtocolError(400, "expected_count must be a non-negative integer")
    return generation, expected


def stats_terms(payload: dict) -> List[str]:
    terms = payload.get("terms")
    if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
        raise ProtocolError(400, "terms must be a list of strings")
    return terms


def search_request(payload: dict) -> dict:
    if not isinstance(payload.get("query"), str):
        raise ProtocolError(400, "query must be a string")
    if "generation" not in payload or not isinstance(payload["generation"], str):
        raise ProtocolError(400, "generation must be a string")
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise ProtocolError(400, "stats must be an object")
    try:
        n_docs = _as_int(stats["n_docs"], "n_docs")
        total_tokens = _as_int(stats["total_tokens"], "total_tokens")
        df = {str(term): _as_int(value, "df") for term, value in dict(stats["df"]).items()}
        top_k = _as_int(payload.get("top_k", 5), "top_k")
        overlap = payload.get("min_overlap", 0.0)
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float)):
            raise ProtocolError(400, "min_overlap must be a number")
        min_overlap = float(overlap)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(400, "search stats are incomplete") from exc
    if n_docs < 0 or total_tokens < 0 or any(value < 0 for value in df.values()):
        raise ProtocolError(400, "search statistics must be non-negative")
    if top_k < 0 or top_k > MAX_TOP_K:
        raise ProtocolError(400, f"top_k must be between 0 and {MAX_TOP_K}")
    company = payload.get("company")
    if company is not None and not isinstance(company, str):
        raise ProtocolError(400, "company must be a string or null")
    return {
        "query": payload["query"],
        "company": company,
        "top_k": top_k,
        "min_overlap": min_overlap,
        "generation": payload["generation"],
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "df": df,
    }


def hit_payload(doc: Document, score: float, overlap: float = 0.0) -> dict:
    return {
        "doc_id": doc.doc_id,
        "score": score,
        "overlap": overlap,
        "company": doc.company,
        "product_area": doc.product_area,
        "text": doc.text,
        "source": str(doc.path),
    }


def dumps(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def chunked(items: Sequence[Document], size: int) -> List[List[Document]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [list(items[start : start + size]) for start in range(0, len(items), size)]

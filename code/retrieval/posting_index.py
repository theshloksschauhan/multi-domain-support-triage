"""BM25 over normalized SQLite postings.

Each term is a row ``(term, doc_id, tf)`` with an index on ``term``. A query
reads those lists and scores with the same formula as ``BM25Index``. This
layout is the relational comparison; the in-memory term-frequency layout
remains the one the shard server loads unless the bake-off says otherwise.
"""

import json
import math
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from data_loader.corpus_loader import Document
from utils.text_cleaning import overlap_ratio, tokenize


class PostingIndex:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                product_area TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                tokens_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS postings (
                term TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (term, doc_id)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS postings_term ON postings(term)")
        self.k1 = 1.5
        self.b = 0.75

    def close(self) -> None:
        self._conn.close()

    def build(self, docs: Sequence[Document]) -> None:
        self._conn.execute("DELETE FROM postings")
        self._conn.execute("DELETE FROM documents")
        for doc in docs:
            self._conn.execute(
                """
                INSERT INTO documents (
                    doc_id, company, product_area, text, source, token_count, tokens_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.company,
                    doc.product_area,
                    doc.text,
                    str(doc.path),
                    len(doc.tokens),
                    json.dumps(list(doc.tokens), ensure_ascii=False),
                ),
            )
            self._conn.executemany(
                "INSERT INTO postings (term, doc_id, tf) VALUES (?, ?, ?)",
                [(term, doc.doc_id, int(tf)) for term, tf in doc.term_freq.items()],
            )
        self._conn.commit()

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 5,
        min_overlap: float = 0.0,
    ) -> List[Tuple[str, float]]:
        query_tokens = tokenize(query)
        if not query_tokens or top_k <= 0:
            return []
        n_docs, total_tokens = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(token_count), 0) AS tokens FROM documents"
        ).fetchone()
        n_docs = int(n_docs)
        if n_docs == 0:
            return []
        avgdl = int(total_tokens) / n_docs
        frequencies = {}
        candidates = set()
        for term in dict.fromkeys(query_tokens):
            rows = self._conn.execute(
                "SELECT doc_id, tf FROM postings WHERE term = ?",
                (term,),
            ).fetchall()
            frequencies[term] = {row["doc_id"]: int(row["tf"]) for row in rows}
            candidates.update(frequencies[term])
        if not candidates:
            return []
        docs = {}
        candidate_ids = list(candidates)
        for start in range(0, len(candidate_ids), 400):
            chunk = candidate_ids[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            for row in self._conn.execute(
                f"""
                SELECT doc_id, company, token_count, tokens_json
                FROM documents WHERE doc_id IN ({placeholders})
                """,
                tuple(chunk),
            ):
                docs[row["doc_id"]] = row
        idf = {
            term: math.log((n_docs - len(doc_tfs) + 0.5) / (len(doc_tfs) + 0.5) + 1)
            for term, doc_tfs in frequencies.items()
        }
        scored: List[Tuple[str, float]] = []
        for doc_id in candidates:
            row = docs[doc_id]
            if company and company != "None" and row["company"] != company:
                continue
            doc_terms = set(json.loads(row["tokens_json"]))
            if min_overlap and overlap_ratio(query_tokens, doc_terms) < min_overlap:
                continue
            score = 0.0
            doc_len = int(row["token_count"])
            for term in query_tokens:
                tf = frequencies[term].get(doc_id, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                score += idf[term] * (tf * (self.k1 + 1) / denom)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

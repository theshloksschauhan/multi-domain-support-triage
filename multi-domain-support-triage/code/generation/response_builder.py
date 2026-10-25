import html as html_lib
import re
from typing import List, Tuple

from config.settings import ESCALATE_RESPONSE, OUT_OF_SCOPE_RESPONSE
from data_loader.corpus_loader import Document
from utils.text_cleaning import split_sentences, tokenize


def select_snippets(text: str, query: str, max_sentences: int) -> List[str]:
    query_tokens = set(tokenize(query))
    sentences = split_sentences(text)
    scored: List[Tuple[int, str]] = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if len(cleaned) < 20:
            continue
        s_tokens = set(tokenize(cleaned))
        overlap = len(s_tokens & query_tokens)
        if overlap > 0:
            scored.append((overlap, cleaned))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if scored:
        return [s for _, s in scored[:max_sentences]]
    return [s for s in sentences[:max_sentences] if s.strip()]


def build_response(
    query: str,
    results: List[Tuple[Document, float]],
    request_type: str,
    max_sentences: int,
) -> str:
    if request_type == "invalid":
        return OUT_OF_SCOPE_RESPONSE
    if not results:
        return ESCALATE_RESPONSE
    best_doc, _score = results[0]
    snippets = select_snippets(best_doc.text, query, max_sentences)
    if not snippets:
        return ESCALATE_RESPONSE
    parts: List[str] = []
    for snippet in snippets:
        cleaned = re.sub(r"^[-*\d.)\s]+", "", snippet).strip()
        if cleaned:
            parts.append(html_lib.unescape(cleaned))
    return " ".join(parts)

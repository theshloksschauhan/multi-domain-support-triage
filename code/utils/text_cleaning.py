import re
from typing import Iterable, List, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in tokens if t and t not in STOPWORDS]


def clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = text.replace("\\", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_sentences: int = 6, overlap: int = 2) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks = []
    step = max(max_sentences - overlap, 1)
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


def split_front_matter(text: str) -> Tuple[str, str]:
    """
    Split YAML front matter from Markdown body.

    Must not use naive ``split('---')``: URLs and titles may contain ``---``
    (for example ``zoom---hackerrank`` inside quoted YAML values).
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", text
    fm_lines: List[str] = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            body = "".join(lines[i + 1 :])
            return "".join(fm_lines), body
        fm_lines.append(lines[i])
        i += 1
    return "", text


def normalize_area(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def overlap_ratio(query_tokens: Iterable[str], doc_terms: Iterable[str]) -> float:
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    doc_set = set(doc_terms)
    return len(query_set & doc_set) / len(query_set)

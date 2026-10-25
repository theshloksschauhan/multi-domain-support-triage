from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

from classification.product_area import infer_company_from_path, infer_product_area_from_path
from utils.front_matter import extract_breadcrumbs
from utils.text_cleaning import clean_markdown, chunk_text, split_front_matter, tokenize


@dataclass(frozen=True)
class Document:
    doc_id: str
    company: str
    product_area: str
    text: str
    tokens: List[str]
    term_freq: Counter
    path: Path


def load_corpus(corpus_root: Path) -> List[Document]:
    docs: List[Document] = []
    if not corpus_root.exists():
        return docs
    paths = sorted(corpus_root.rglob("*.md"), key=lambda p: str(p).lower())
    for path in paths:
        if path.name.lower() == "index.md":
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        fm, body = split_front_matter(raw)
        company = infer_company_from_path(path)
        crumbs = extract_breadcrumbs(fm)
        product_area = (
            " · ".join(crumbs) if crumbs else infer_product_area_from_path(path)
        )
        cleaned = clean_markdown(body)
        for idx, chunk in enumerate(chunk_text(cleaned)):
            tokens = tokenize(chunk)
            if not tokens:
                continue
            doc_id = f"{path.relative_to(corpus_root)}::chunk{idx}"
            docs.append(
                Document(
                    doc_id=doc_id,
                    company=company,
                    product_area=product_area,
                    text=chunk,
                    tokens=tokens,
                    term_freq=Counter(tokens),
                    path=path,
                )
            )
    return docs

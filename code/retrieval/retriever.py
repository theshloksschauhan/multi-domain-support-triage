from typing import List, Optional, Tuple

from data_loader.corpus_loader import Document
from retrieval.bm25 import BM25Index


class Retriever:
    def __init__(self, index: BM25Index, min_overlap: float, top_k: int):
        self.index = index
        self.min_overlap = min_overlap
        self.top_k = top_k

    def search(self, query: str, company: Optional[str]) -> List[Tuple[Document, float]]:
        return self.index.search(
            query,
            company=company,
            top_k=self.top_k,
            min_overlap=self.min_overlap,
        )

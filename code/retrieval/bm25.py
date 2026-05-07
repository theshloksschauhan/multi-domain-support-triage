import math
from collections import Counter
from typing import List, Optional, Tuple

from data_loader.corpus_loader import Document
from utils.text_cleaning import overlap_ratio, tokenize


class BM25Index:
    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(d.tokens) for d in docs) / max(len(docs), 1)
        self.doc_freq = Counter()
        for doc in docs:
            for term in set(doc.tokens):
                self.doc_freq[term] += 1
        self.idf = {
            term: math.log((len(docs) - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freq.items()
        }

    def score(self, query_tokens: List[str], doc: Document) -> float:
        score = 0.0
        doc_len = len(doc.tokens)
        for term in query_tokens:
            if term not in doc.term_freq:
                continue
            tf = doc.term_freq[term]
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (tf * (self.k1 + 1) / denom)
        return score

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 5,
        min_overlap: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        results: List[Tuple[Document, float]] = []
        for doc in self.docs:
            if company and company != "None" and doc.company != company:
                continue
            if min_overlap and overlap_ratio(query_tokens, doc.term_freq.keys()) < min_overlap:
                continue
            score = self.score(query_tokens, doc)
            if score > 0:
                results.append((doc, score))
        results.sort(key=lambda x: (-x[1], x[0].doc_id))
        return results[:top_k]

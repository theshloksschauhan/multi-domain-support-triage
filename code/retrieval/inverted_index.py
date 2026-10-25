"""Inverted-index BM25 search with the same results as ``BM25Index``.

Postings map each term to the documents whose ``term_freq`` contains it.
Ordinary queries score that candidate set instead of scanning every document.

Scores still use the full-scan formula and the corpus-wide statistics from
``BM25Index``: ``k1=1.5``, ``b=0.75``, and

    idf = log((N - df + 0.5) / (df + 0.5) + 1)

Document frequency is counted from ``set(doc.tokens)``, exactly as the
reference does. Candidate lookup uses ``term_freq`` because that is what
``BM25Index.score`` reads. Those two maps can disagree on a hand-built
document, so postings follow ``term_freq`` and idf follows the token sets.

``BM25Index.search`` keeps a hit only when its score is strictly greater
than zero. A term that occurs in at least one token set has positive idf, so
a document can score above zero only when a query token is in its
``term_freq``. Documents outside the postings cannot appear in the reference
results, including when ``min_overlap`` is 0 and ``top_k`` is larger than the
number of positive hits. Zero-score documents are therefore omitted here as
well, with no full-corpus fallback.

Repeated query tokens are scored once per occurrence. ``company="None"`` does
not filter. A falsy ``min_overlap``, including ``0``, does not filter.
Equal scores are ordered by ascending ``doc_id``. If that still ties, the
original document position is used, which is what the reference's stable sort
does. Document ID strings are returned unchanged.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from data_loader.corpus_loader import Document
from utils.text_cleaning import overlap_ratio, tokenize


def bm25_idf(n_docs: int, df: int) -> float:
    """IDF used by ``BM25Index``: ``log((N - df + 0.5) / (df + 0.5) + 1)``."""
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1)


@dataclass(frozen=True)
class GlobalBm25Stats:
    """Corpus-wide statistics shared by every shard for one query.

    ``df`` is the sum of per-shard document frequencies. A term with no
    recorded df scores as zero, matching ``BM25Index`` when the term is
    absent from the idf table. Statistics are not restricted to a company.
    """

    n_docs: int
    total_tokens: int
    df: Mapping[str, int]

    @property
    def avgdl(self) -> float:
        return self.total_tokens / max(self.n_docs, 1)

    def idf(self, term: str) -> float:
        df = int(self.df.get(term, 0))
        if df <= 0 or self.n_docs <= 0:
            return 0.0
        return bm25_idf(self.n_docs, df)


class InvertedIndex:
    """BM25 search over a term-to-document postings map."""

    def __init__(self, docs: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.total_tokens = sum(len(doc.tokens) for doc in docs)
        self.avgdl = self.total_tokens / max(len(docs), 1)
        self.doc_lengths = [len(doc.tokens) for doc in docs]
        self.doc_freq: Counter[str] = Counter()
        mutable_postings: Dict[str, List[int]] = {}
        for index, doc in enumerate(docs):
            for term in set(doc.tokens):
                self.doc_freq[term] += 1
            for term in doc.term_freq:
                mutable_postings.setdefault(term, []).append(index)
        self.postings: Dict[str, Tuple[int, ...]] = {
            term: tuple(indexes) for term, indexes in mutable_postings.items()
        }
        n_docs = len(docs)
        self.idf = {
            term: bm25_idf(n_docs, df) for term, df in self.doc_freq.items()
        }

    @property
    def document_count(self) -> int:
        return len(self.docs)

    def score(
        self,
        query_tokens: List[str],
        doc: Document,
        stats: Optional[GlobalBm25Stats] = None,
    ) -> float:
        """Same arithmetic, in the same order, as ``BM25Index.score``.

        ``stats`` replaces this index's ``idf`` and ``avgdl`` when a
        coordinator has summed those values across shards. Omitting it keeps
        the single-index behavior.
        """
        score = 0.0
        doc_len = len(doc.tokens)
        avgdl = self.avgdl if stats is None else stats.avgdl
        for term in query_tokens:
            if term not in doc.term_freq:
                continue
            tf = doc.term_freq[term]
            if stats is None:
                idf = self.idf.get(term, 0.0)
            else:
                idf = stats.idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            score += idf * (tf * (self.k1 + 1) / denom)
        return score

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 5,
        min_overlap: float = 0.0,
        stats: Optional[GlobalBm25Stats] = None,
    ) -> List[Tuple[Document, float]]:
        """Score candidates from the posting lists of the query terms.

        Building the lists is linear in the number of tokens in the corpus.
        One query is linear in the length of those posting lists. Merging
        shard results is linear in the number of partitions after each
        partition has returned its own top hits.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        candidates = set()
        for term in query_tokens:
            candidates.update(self.postings.get(term, ()))
        results: List[Tuple[Document, float, int]] = []
        for index in candidates:
            doc = self.docs[index]
            if company and company != "None" and doc.company != company:
                continue
            if min_overlap and overlap_ratio(query_tokens, doc.term_freq.keys()) < min_overlap:
                continue
            scored = self.score(query_tokens, doc, stats)
            if scored > 0:
                results.append((doc, scored, index))
        results.sort(key=lambda item: (-item[1], item[0].doc_id, item[2]))
        return [(doc, score) for doc, score, _index in results[:top_k]]

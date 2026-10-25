"""Parity tests: InvertedIndex must match BM25Index on the same documents.

Score comparisons use an absolute tolerance of 1e-12. The two implementations
perform the same floating-point operations in the same order, so real
differences are algorithmic rather than rounding noise.
"""

import math
import sys
import time
import unittest
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from config.settings import MIN_OVERLAP, repo_root
from data_loader.corpus_loader import Document, load_corpus
from data_loader.ticket_loader import load_tickets
from retrieval.backend import build_search_backend
from retrieval.bm25 import BM25Index
from retrieval.inverted_index import InvertedIndex
from utils.text_cleaning import tokenize

SCORE_ABS_TOLERANCE = 1e-12


def _doc(doc_id: str, text: str, company: str = "HackerRank") -> Document:
    tokens = tokenize(text)
    return Document(
        doc_id=doc_id,
        company=company,
        product_area="general",
        text=text,
        tokens=tokens,
        term_freq=Counter(tokens),
        path=Path("fixture.md"),
    )


def _fixture() -> List[Document]:
    long_text = "shared " + " ".join(f"filler{i}" for i in range(40))
    return [
        _doc("b-doc", "shared alpha beta gamma"),
        _doc("a-doc", "shared alpha beta gamma"),
        _doc("short", "shared raretoken"),
        _doc("medium", "shared alpha beta gamma delta epsilon zeta eta theta"),
        _doc("long", long_text, company="Visa"),
        _doc("visa-doc", "shared visa dispute charge issuing bank", company="Visa"),
        _doc(r"claude\path\file.md::chunk0", "shared claude privacy crawler", company="Claude"),
    ]


class InvertedIndexParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = _fixture()
        self.reference = BM25Index(self.docs)
        self.inverted = InvertedIndex(self.docs)

    def assert_parity(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 5,
        min_overlap: float = 0.0,
        docs: Optional[Sequence[Document]] = None,
    ) -> Tuple[list, list]:
        if docs is None:
            reference, inverted = self.reference, self.inverted
        else:
            reference, inverted = BM25Index(docs), InvertedIndex(docs)
        left = reference.search(query, company=company, top_k=top_k, min_overlap=min_overlap)
        right = inverted.search(query, company=company, top_k=top_k, min_overlap=min_overlap)
        left_ids = [doc.doc_id for doc, _score in left]
        right_ids = [doc.doc_id for doc, _score in right]
        self.assertEqual(left_ids, right_ids, msg=f"query={query!r} company={company!r}")
        self.assertEqual(len(left), len(right))
        for (left_doc, left_score), (right_doc, right_score) in zip(left, right):
            self.assertIs(left_doc, right_doc)
            self.assertTrue(
                math.isclose(left_score, right_score, rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE),
                f"{left_score} != {right_score} for {left_doc.doc_id}",
            )
        return left, right

    def test_statistics_match_reference(self) -> None:
        self.assertEqual(self.inverted.k1, 1.5)
        self.assertEqual(self.inverted.b, 0.75)
        self.assertEqual(self.inverted.document_count, len(self.docs))
        self.assertEqual(self.inverted.total_tokens, sum(len(doc.tokens) for doc in self.docs))
        self.assertEqual(self.inverted.avgdl, self.reference.avgdl)
        self.assertEqual(self.inverted.doc_freq, self.reference.doc_freq)
        self.assertEqual(self.inverted.idf, self.reference.idf)
        self.assertEqual(self.inverted.doc_freq["shared"], len(self.docs))
        self.assertEqual(self.inverted.doc_freq["raretoken"], 1)
        self.assertEqual(self.inverted.postings["raretoken"], (2,))
        self.assertEqual(len(self.inverted.postings["shared"]), len(self.docs))

    def test_empty_corpus(self) -> None:
        reference = BM25Index([])
        inverted = InvertedIndex([])
        self.assertEqual(inverted.document_count, 0)
        self.assertEqual(inverted.total_tokens, 0)
        self.assertEqual(inverted.avgdl, reference.avgdl)
        self.assertEqual(inverted.doc_freq, reference.doc_freq)
        self.assertEqual(inverted.search("alpha"), [])
        self.assertEqual(reference.search("alpha"), [])

    def test_empty_unknown_and_stopword_queries(self) -> None:
        for query in ("", "zzzznotaterm999", "the and of"):
            left, right = self.assert_parity(query, top_k=20)
            self.assertEqual(right, [])

    def test_single_term_multiple_terms_and_repeats(self) -> None:
        self.assert_parity("alpha")
        self.assert_parity("alpha beta gamma")
        self.assert_parity("alpha alpha")
        once = self.inverted.search("alpha", top_k=1)[0][1]
        twice = self.inverted.search("alpha alpha", top_k=1)[0][1]
        reference_twice = self.reference.search("alpha alpha", top_k=1)[0][1]
        self.assertTrue(math.isclose(twice, once * 2, rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE))
        self.assertTrue(math.isclose(twice, reference_twice, rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE))

    def test_zero_overlap_documents_are_excluded(self) -> None:
        left, right = self.assert_parity("alpha", top_k=100, min_overlap=0.0)
        ids = [doc.doc_id for doc, _score in right]
        self.assertNotIn("short", ids)
        self.assertLess(len(right), len(self.docs))
        self.assertGreater(len(right), 0)
        self.assertEqual([doc.doc_id for doc, _score in left], ids)

    def test_top_k_limits(self) -> None:
        one, _ = self.assert_parity("shared", top_k=1)
        self.assertEqual(len(one), 1)
        all_hits, _ = self.assert_parity("shared", top_k=100)
        self.assertEqual(len(all_hits), len(self.docs))
        self.assert_parity("raretoken", top_k=100)

    def test_exact_ties_follow_doc_id_then_position(self) -> None:
        left, right = self.assert_parity("alpha beta gamma", top_k=2)
        self.assertEqual([doc.doc_id for doc, _score in right], ["a-doc", "b-doc"])
        self.assertTrue(math.isclose(right[0][1], right[1][1], rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE))
        duplicate_id = [
            _doc("same", "alpha beta gamma"),
            _doc("same", "alpha beta gamma extra"),
            _doc("same", "alpha beta gamma"),
        ]
        tied, _ = self.assert_parity("alpha beta gamma", top_k=10, docs=duplicate_id)
        self.assertIs(tied[0][0], duplicate_id[0])
        self.assertIs(tied[1][0], duplicate_id[2])

    def test_company_filters(self) -> None:
        visa, _ = self.assert_parity("shared dispute", company="Visa", top_k=10)
        self.assertTrue(visa)
        self.assertTrue(all(doc.company == "Visa" for doc, _score in visa))
        missing, _ = self.assert_parity("shared", company="MissingCo", top_k=10)
        self.assertEqual(missing, [])
        unfiltered, _ = self.assert_parity("shared dispute", company=None, top_k=10)
        none_company, _ = self.assert_parity("shared dispute", company="None", top_k=10)
        self.assertEqual(
            [doc.doc_id for doc, _score in unfiltered],
            [doc.doc_id for doc, _score in none_company],
        )
        self.assertGreater(len(unfiltered), len(visa))

    def test_overlap_thresholds(self) -> None:
        query = "alpha " + " ".join(f"unk{i}" for i in range(20))
        kept, _ = self.assert_parity(query, top_k=20, min_overlap=0.0)
        self.assertTrue(kept)
        dropped, _ = self.assert_parity(query, top_k=20, min_overlap=MIN_OVERLAP)
        self.assertEqual(dropped, [])
        strict, _ = self.assert_parity("alpha notinthedocs", top_k=20, min_overlap=0.9)
        self.assertEqual(strict, [])
        kept, _ = self.assert_parity("alpha beta", top_k=20, min_overlap=1.0)
        self.assertTrue(kept)
        self.assertTrue(all("alpha" in doc.term_freq and "beta" in doc.term_freq for doc, _score in kept))

    def test_different_lengths_and_windows_doc_id(self) -> None:
        hits, _ = self.assert_parity("shared", top_k=10)
        lengths = {doc.doc_id: len(doc.tokens) for doc, _score in hits}
        self.assertGreater(len(set(lengths.values())), 1)
        windows, _ = self.assert_parity("claude privacy", top_k=5)
        self.assertEqual(windows[0][0].doc_id, r"claude\path\file.md::chunk0")

    def test_term_freq_and_tokens_can_diverge(self) -> None:
        alpha_tokens = tokenize("alpha beta")
        other_tokens = tokenize("beta gamma")
        source = Document(
            "source",
            "HackerRank",
            "general",
            "alpha beta",
            alpha_tokens,
            Counter(alpha_tokens),
            Path("source.md"),
        )
        extra = Counter(other_tokens)
        extra["alpha"] = 2
        only_freq = Document(
            "only-freq",
            "HackerRank",
            "general",
            "beta gamma",
            other_tokens,
            extra,
            Path("freq.md"),
        )
        only_tokens = Document(
            "only-tokens",
            "Visa",
            "general",
            "alpha beta",
            alpha_tokens,
            Counter(tokenize("beta")),
            Path("tokens.md"),
        )
        docs = [source, only_freq, only_tokens]
        hits, _ = self.assert_parity("alpha", top_k=10, docs=docs)
        ids = [doc.doc_id for doc, _score in hits]
        self.assertEqual(ids, ["only-freq", "source"])
        self.assertEqual(InvertedIndex(docs).doc_freq["alpha"], 2)

    def test_rebuild_is_identical_and_search_does_not_mutate(self) -> None:
        rebuilt = InvertedIndex(self.docs)
        self.assertEqual(rebuilt.doc_freq, self.inverted.doc_freq)
        self.assertEqual(rebuilt.idf, self.inverted.idf)
        self.assertEqual(rebuilt.postings, self.inverted.postings)
        self.assertEqual(rebuilt.avgdl, self.inverted.avgdl)
        self.assertEqual(rebuilt.total_tokens, self.inverted.total_tokens)
        before = self._snapshot(self.inverted)
        for query in ("alpha", "alpha alpha", "raretoken", "shared", "zzzz", ""):
            self.inverted.search(query, company="Visa", top_k=3, min_overlap=MIN_OVERLAP)
            self.inverted.search(query, company=None, top_k=20, min_overlap=0.0)
        self.assertEqual(before, self._snapshot(self.inverted))
        self.assert_parity("alpha beta", top_k=10)

    def test_default_backend_remains_full_scan(self) -> None:
        backend = build_search_backend("local", self.docs, 0.0, 5)
        self.assertIsNotNone(backend)
        assert backend is not None
        self.assertIsInstance(backend.retriever.index, BM25Index)

    def _snapshot(self, index: InvertedIndex) -> tuple:
        return (
            index.avgdl,
            index.total_tokens,
            index.document_count,
            tuple(sorted(index.doc_freq.items())),
            tuple(sorted(index.idf.items())),
            tuple(sorted((term, indexes) for term, indexes in index.postings.items())),
            tuple(doc.doc_id for doc in index.docs),
            tuple(tuple(sorted(doc.term_freq.items())) for doc in index.docs),
            tuple(index.doc_lengths),
        )


class CorpusParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus_root = repo_root() / "data"
        tickets_path = repo_root() / "support_tickets" / "support_tickets.csv"
        if not corpus_root.exists() or not tickets_path.exists():
            raise unittest.SkipTest("local corpus or tickets are not available")
        cls.docs = load_corpus(corpus_root)
        cls.tickets = load_tickets(tickets_path)
        if not cls.docs or not cls.tickets:
            raise unittest.SkipTest("local corpus or tickets are empty")
        cls.reference = BM25Index(cls.docs)
        cls.inverted = InvertedIndex(cls.docs)

    def assert_same(
        self,
        query: str,
        company: Optional[str],
        top_k: int,
        min_overlap: float,
    ) -> None:
        left = self.reference.search(query, company=company, top_k=top_k, min_overlap=min_overlap)
        right = self.inverted.search(query, company=company, top_k=top_k, min_overlap=min_overlap)
        self.assertEqual([doc.doc_id for doc, _score in left], [doc.doc_id for doc, _score in right])
        self.assertEqual(len(left), len(right))
        for (_left_doc, left_score), (_right_doc, right_score) in zip(left, right):
            self.assertTrue(
                math.isclose(left_score, right_score, rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE)
            )

    def test_corpus_statistics_and_queries(self) -> None:
        self.assertEqual(self.inverted.document_count, len(self.docs))
        self.assertEqual(self.inverted.avgdl, self.reference.avgdl)
        self.assertEqual(self.inverted.doc_freq, self.reference.doc_freq)
        self.assertEqual(self.inverted.idf, self.reference.idf)
        self.assertGreater(self.inverted.document_count, 1000)

        queries: List[Tuple[str, Optional[str], int, float]] = [
            ("", None, 5, 0.0),
            ("zzzznotaterm999", None, 5, 0.0),
            ("visa visa dispute", "Visa", 5, MIN_OVERLAP),
            ("visa visa dispute", None, 1, 0.0),
            ("assessment candidate test", "HackerRank", 5, MIN_OVERLAP),
            ("claude privacy crawler", "Claude", 20, 0.0),
            ("the and of", "None", 5, MIN_OVERLAP),
        ]
        rare = next(term for term, df in self.reference.doc_freq.items() if df == 1)
        common = max(self.reference.doc_freq, key=self.reference.doc_freq.get)
        queries.append((rare, None, 5, 0.0))
        queries.append((common, None, 5, MIN_OVERLAP))
        queries.append((f"{common} {common}", None, 10, 0.0))
        for ticket in self.tickets:
            query = f"{ticket.subject} {ticket.issue}".strip()
            queries.append((query, ticket.company or None, 5, MIN_OVERLAP))
            queries.append((query, "None", 5, 0.0))
        queries.append(("dispute charge", "Visa", len(self.docs), 0.0))
        samples = []
        for query, company, top_k, overlap in queries:
            self.assert_same(query, company, top_k, overlap)
            started = time.perf_counter()
            self.inverted.search(query, company=company, top_k=top_k, min_overlap=overlap)
            samples.append(time.perf_counter() - started)
        ordered = sorted(samples)
        position = (len(ordered) - 1) * 0.95
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            p95 = ordered[lower]
        else:
            weight = position - lower
            p95 = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        self.assertLess(p95, 0.200, f"query p95 {p95:.3f}s exceeded the 200ms guard")


if __name__ == "__main__":
    unittest.main()

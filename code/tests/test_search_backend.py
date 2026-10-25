"""Compatibility tests for the local search backend.

These lock the current BM25 search contract and check that routing the
pipeline through the backend leaves the CSV schema and ticket decisions intact.
"""

import csv
import logging
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import List

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from config.settings import (
    DEFAULT_SEARCH_MODE,
    DISTRIBUTED_SEARCH_MODE,
    RuntimeConfig,
    default_config,
)
from data_loader.corpus_loader import Document
from main import build_arg_parser, build_runtime_config
from output.writer import FIELDNAMES
from pipeline.run_pipeline import run_pipeline
from retrieval.backend import LocalSearchBackend, build_search_backend
from retrieval.distributed_retriever import RetrievalUnavailable
from retrieval.bm25 import BM25Index
from retrieval.retriever import Retriever
from utils.text_cleaning import tokenize


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


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("triage.test.search_backend")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class SearchBackendCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            _doc("b-doc", "Alpha beta gamma appears in the hackerrank assessment guide."),
            _doc("a-doc", "Alpha beta gamma appears in the hackerrank assessment guide."),
            _doc("visa-doc", "Visa cardholders can dispute a charge with the issuing bank.", "Visa"),
        ]
        self.index = BM25Index(self.docs)
        self.backend = LocalSearchBackend(self.index, min_overlap=0.0, top_k=5)
        self.retriever = Retriever(self.index, min_overlap=0.0, top_k=5)

    def _ids(self, results: List[tuple]) -> List[str]:
        return [doc.doc_id for doc, _score in results]

    def test_local_backend_matches_existing_search(self) -> None:
        queries = [
            ("", None, 0.0),
            ("zzzznotaterm999", None, 0.0),
            ("alpha beta", None, 0.0),
            ("alpha alpha beta", None, 0.0),
            ("dispute charge", "Visa", 0.0),
            ("dispute charge", "None", 0.0),
            ("alpha beta gamma extra", None, 0.9),
        ]
        for query, company, overlap in queries:
            direct = self.index.search(query, company=company, top_k=5, min_overlap=overlap)
            via_retriever = Retriever(self.index, overlap, 5).search(query, company)
            via_backend = LocalSearchBackend(self.index, overlap, 5).search(query, company)
            self.assertEqual(self._ids(direct), self._ids(via_retriever))
            self.assertEqual(self._ids(direct), self._ids(via_backend))
            self.assertEqual([score for _doc, score in direct], [score for _doc, score in via_backend])

    def test_bm25_parameters_and_idf_formula(self) -> None:
        self.assertEqual(self.index.k1, 1.5)
        self.assertEqual(self.index.b, 0.75)
        self.assertAlmostEqual(
            self.index.avgdl,
            sum(len(doc.tokens) for doc in self.docs) / len(self.docs),
        )
        term = "alpha"
        df = self.index.doc_freq[term]
        n_docs = len(self.docs)
        expected = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        self.assertAlmostEqual(self.index.idf[term], expected)

    def test_empty_and_unknown_queries_return_nothing(self) -> None:
        self.assertEqual(self.backend.search("", None), [])
        self.assertEqual(self.backend.search("zzzznotaterm999", None), [])

    def test_duplicate_query_terms_are_scored_again(self) -> None:
        once = self.index.search("alpha", top_k=1, min_overlap=0.0)[0][1]
        twice = self.index.search("alpha alpha", top_k=1, min_overlap=0.0)[0][1]
        self.assertAlmostEqual(twice, once * 2)

    def test_tie_breaks_on_doc_id(self) -> None:
        results = self.backend.search("alpha beta gamma", None)
        self.assertEqual(self._ids(results[:2]), ["a-doc", "b-doc"])
        self.assertAlmostEqual(results[0][1], results[1][1])

    def test_company_filter_preserves_none_sentinel(self) -> None:
        filtered = self.backend.search("alpha dispute", "Visa")
        self.assertEqual(self._ids(filtered), ["visa-doc"])
        unfiltered = self.backend.search("alpha dispute", "None")
        self.assertIn("a-doc", self._ids(unfiltered))
        self.assertIn("visa-doc", self._ids(unfiltered))

    def test_factory_local_mode_and_empty_corpus(self) -> None:
        built = build_search_backend(DEFAULT_SEARCH_MODE, self.docs, 0.0, 5)
        self.assertIsInstance(built, LocalSearchBackend)
        self.assertIsNone(build_search_backend(DEFAULT_SEARCH_MODE, [], 0.0, 5))

    def test_distributed_mode_does_not_fall_back(self) -> None:
        with self.assertRaises(RetrievalUnavailable):
            build_search_backend(DISTRIBUTED_SEARCH_MODE, self.docs, 0.0, 5)
        with self.assertRaises(RetrievalUnavailable):
            build_search_backend(DISTRIBUTED_SEARCH_MODE, [], 0.0, 5)

    def test_default_cli_search_mode_is_local(self) -> None:
        self.assertEqual(default_config().search_mode, DEFAULT_SEARCH_MODE)
        args = build_arg_parser(default_config()).parse_args([])
        config = build_runtime_config(args)
        self.assertEqual(config.search_mode, DEFAULT_SEARCH_MODE)

    def test_pipeline_keeps_csv_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "data" / "visa"
            corpus.mkdir(parents=True)
            (corpus / "support.md").write_text(
                "Visa cardholders can dispute a charge by contacting the issuing bank. "
                "The dispute form asks for the transaction date and the amount.\n",
                encoding="utf-8",
            )
            tickets = root / "tickets.csv"
            tickets.write_text(
                "Issue,Subject,Company\n"
                "How do I dispute a charge on my visa card,Dispute,Visa\n"
                "Please restore my access even though I am not the workspace owner,Access,Claude\n",
                encoding="utf-8",
            )
            first = root / "first.csv"
            second = root / "second.csv"
            log_path = root / "log.txt"

            def run(output: Path) -> None:
                config = RuntimeConfig(
                    input_path=tickets,
                    output_path=output,
                    corpus_root=root / "data",
                    sample_path=tickets,
                    log_path=log_path,
                    min_score=0.01,
                    min_overlap=0.0,
                    top_k=5,
                    max_sentences=4,
                    search_mode=DEFAULT_SEARCH_MODE,
                )
                run_pipeline(config, _quiet_logger())

            run(first)
            run(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            with first.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader.fieldnames or []), FIELDNAMES)
                rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "replied")
        self.assertIn("dispute", rows[0]["response"].lower())
        self.assertEqual(rows[1]["status"], "escalated")
        self.assertIn("Escalation rule matched", rows[1]["justification"])
        allowed = {"product_issue", "bug", "feature_request", "invalid"}
        self.assertTrue(all(row["request_type"] in allowed for row in rows))


if __name__ == "__main__":
    unittest.main()

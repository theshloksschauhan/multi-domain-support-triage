import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data_loader.corpus_loader import Document
from retrieval.bm25 import BM25Index
from retrieval.posting_index import PostingIndex
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
        path=Path(f"{doc_id}.md"),
    )


class PostingIndexTests(unittest.TestCase):
    def test_scores_match_bm25(self) -> None:
        docs = [
            _doc("b-doc", "shared alpha beta gamma"),
            _doc("a-doc", "shared alpha beta gamma"),
            _doc("short", "shared raretoken"),
            _doc("visa-doc", "shared visa dispute charge", "Visa"),
        ]
        reference = BM25Index(docs)
        tmp = tempfile.TemporaryDirectory()
        store = PostingIndex(Path(tmp.name) / "postings.db")
        try:
            store.build(docs)
            for query, company, overlap in (
                ("alpha beta", None, 0.0),
                ("alpha alpha", None, 0.0),
                ("visa dispute", "Visa", 0.0),
                ("shared", "None", 0.5),
                ("", None, 0.0),
            ):
                left = reference.search(query, company=company, top_k=5, min_overlap=overlap)
                right = store.search(query, company=company, top_k=5, min_overlap=overlap)
                self.assertEqual([doc.doc_id for doc, _score in left], [doc_id for doc_id, _score in right])
                for (_doc_left, left_score), (_doc_id, right_score) in zip(left, right):
                    self.assertAlmostEqual(left_score, right_score, delta=1e-12)
        finally:
            store.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()

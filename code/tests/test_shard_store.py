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
from retrieval.inverted_index import GlobalBm25Stats, InvertedIndex
from retrieval.shard_store import ShardStore, StoreError
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
        path=Path("fixture") / f"{doc_id}.md",
    )


class ShardStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ShardStore(Path(self.tmp.name) / "shard.db")
        self.docs = [
            _doc("b-doc", "alpha beta shared"),
            _doc("a-doc", "alpha beta shared"),
            _doc(r"visa\support.md::chunk0", "visa dispute shared", "Visa"),
        ]

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_replace_reload_matches_bm25(self) -> None:
        self.store.replace_partition(self.docs, "gen-1")
        meta = self.store.metadata()
        self.assertTrue(meta["ready"])
        self.assertEqual(meta["generation"], "gen-1")
        self.assertEqual(meta["document_count"], 3)
        reloaded = ShardStore(self.store.path)
        try:
            index = reloaded.load_index()
            reference = BM25Index(self.docs)
            self.assertEqual(index.document_count, 3)
            self.assertEqual(index.doc_freq, reference.doc_freq)
            self.assertEqual(
                [doc.doc_id for doc, _score in index.search("alpha beta", top_k=5)],
                [doc.doc_id for doc, _score in reference.search("alpha beta", top_k=5)],
            )
            self.assertEqual(index.search("alpha", top_k=1)[0][0].doc_id, r"a-doc")
            windows = index.search("visa dispute", company="Visa", top_k=1)
            self.assertEqual(windows[0][0].doc_id, r"visa\support.md::chunk0")
        finally:
            reloaded.close()

    def test_upsert_is_idempotent_and_update_drops_old_postings(self) -> None:
        self.store.upsert_documents(self.docs)
        self.store.upsert_documents(self.docs)
        self.assertFalse(self.store.metadata()["ready"])
        self.assertEqual(self.store.metadata()["document_count"], 3)
        updated = _doc("a-doc", "gamma only")
        self.store.upsert_documents([updated])
        index = self.store.load_index()
        self.assertEqual(index.search("alpha", top_k=10)[0][0].doc_id, "b-doc")
        self.assertNotIn("a-doc", [doc.doc_id for doc, _score in index.search("alpha", top_k=10)])
        self.assertEqual(index.search("gamma", top_k=1)[0][0].doc_id, "a-doc")

    def test_delete_removes_postings(self) -> None:
        self.store.replace_partition(self.docs, "gen-1")
        self.store.delete_document("b-doc")
        self.assertFalse(self.store.metadata()["ready"])
        index = self.store.load_index()
        ids = [doc.doc_id for doc, _score in index.search("shared", top_k=10)]
        self.assertNotIn("b-doc", ids)
        self.assertEqual(self.store.metadata()["document_count"], 2)

    def test_failed_replace_keeps_previous_generation(self) -> None:
        self.store.replace_partition(self.docs, "gen-1")
        duplicate = [_doc("same", "alpha"), _doc("same", "beta gamma")]
        with self.assertRaises(StoreError):
            self.store.replace_partition(duplicate, "gen-2")
        self.assertEqual(self.store.metadata()["generation"], "gen-1")
        self.assertEqual(self.store.metadata()["document_count"], 3)
        self.assertEqual(self.store.document_ids(), sorted(doc.doc_id for doc in self.docs))

    def test_commit_staged_swaps_atomically(self) -> None:
        self.store.replace_partition(self.docs[:1], "gen-1")
        self.store.clear_staging()
        self.store.stage_documents(self.docs)
        with self.assertRaises(StoreError):
            self.store.commit_staged("gen-2", expected_count=1)
        self.assertEqual(self.store.metadata()["generation"], "gen-1")
        self.assertEqual(self.store.document_ids(), [self.docs[0].doc_id])
        meta = self.store.commit_staged("gen-2", expected_count=len(self.docs))
        self.assertEqual(meta["generation"], "gen-2")
        self.assertEqual(meta["document_count"], len(self.docs))
        self.assertTrue(meta["ready"])

    def test_global_stats_match_local_when_they_describe_the_same_corpus(self) -> None:
        index = InvertedIndex(self.docs)
        stats = GlobalBm25Stats(
            n_docs=index.document_count,
            total_tokens=index.total_tokens,
            df=dict(index.doc_freq),
        )
        local = index.search("alpha beta", top_k=5)
        remote = index.search("alpha beta", top_k=5, stats=stats)
        self.assertEqual(
            [doc.doc_id for doc, _score in local],
            [doc.doc_id for doc, _score in remote],
        )
        self.assertEqual([score for _doc, score in local], [score for _doc, score in remote])


if __name__ == "__main__":
    unittest.main()

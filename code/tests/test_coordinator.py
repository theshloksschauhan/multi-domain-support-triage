import math
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data_loader.corpus_loader import Document
from retrieval.bm25 import BM25Index
from retrieval.distributed_retriever import DistributedRetriever, RetrievalUnavailable
from retrieval.index_coordinator import IndexCoordinator
from retrieval.protocol import DEFAULT_BODY_LIMIT
from retrieval.shard_client import ShardClient
from retrieval.shard_server import ShardHTTPServer, ShardState, make_handler
from retrieval.shard_store import ShardStore
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
        path=Path("docs") / f"{doc_id}.md",
    )


def _fixture() -> list:
    return [
        _doc("b-doc", "shared alpha beta gamma"),
        _doc("a-doc", "shared alpha beta gamma"),
        _doc("short", "shared raretoken"),
        _doc("visa-doc", "shared visa dispute charge", "Visa"),
        _doc(r"claude\path\file.md::chunk0", "shared claude privacy", "Claude"),
    ]


class RunningShard:
    def __init__(self, root: Path, shard_id: str):
        self.store = ShardStore(root / f"{shard_id}.db")
        self.state = ShardState(self.store, shard_id, DEFAULT_BODY_LIMIT)
        self.server = ShardHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        port = self.server.server_address[1]
        self.client = ShardClient(shard_id, f"http://127.0.0.1:{port}", timeout=5.0)
        last_error = None
        for _ in range(50):
            try:
                self.client.health()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        raise RuntimeError(f"{shard_id} did not start: {last_error}")

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.store.close()


class CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.docs = _fixture()
        corpus = self.root / "corpus" / "visa"
        corpus.mkdir(parents=True)
        (corpus / "support.md").write_text(
            "Visa cardholders can dispute a charge with the issuing bank. "
            "Keep the receipt before you submit the dispute.\n",
            encoding="utf-8",
        )
        self.shards = [RunningShard(self.root, f"shard-{index}") for index in range(3)]
        self.clients = [shard.client for shard in self.shards]

    def tearDown(self) -> None:
        for shard in self.shards:
            shard.close()
        self.tmp.cleanup()

    def test_http_roundtrip_restart_and_multi_shard_parity(self) -> None:
        report = IndexCoordinator(self.clients).build(self.root / "corpus")
        self.assertEqual(sum(report["per_shard"].values()), report["documents"])
        self.assertGreater(report["documents"], 0)
        restarted_store = ShardStore(self.shards[0].store.path)
        try:
            self.assertTrue(restarted_store.metadata()["ready"])
            self.assertEqual(restarted_store.metadata()["generation"], report["generation"])
        finally:
            restarted_store.close()

        hand = _fixture()
        for shard in self.shards:
            shard.client.begin_index()
        groups = {client.shard_id: [] for client in self.clients}
        from retrieval.partitioner import assign_shard

        for doc in hand:
            groups[self.clients[assign_shard(doc.doc_id, 3)].shard_id].append(doc)
        from retrieval.partitioner import index_generation

        generation = index_generation(
            [doc.doc_id for doc in hand],
            [len(doc.tokens) for doc in hand],
            [client.shard_id for client in self.clients],
        )
        for client in self.clients:
            from retrieval.protocol import document_to_payload

            assigned = groups[client.shard_id]
            if assigned:
                client.index_batch([document_to_payload(doc) for doc in assigned])
            client.commit(generation, len(assigned))

        reference = BM25Index(hand)
        retriever = DistributedRetriever(self.clients, top_k=5, min_overlap=0.0, timeout=5.0)
        for query, company in (
            ("alpha beta", None),
            ("alpha alpha", None),
            ("raretoken", "None"),
            ("visa dispute", "Visa"),
            ("", None),
            ("zzzznotaterm999", None),
            ("shared", "MissingCo"),
        ):
            left = reference.search(query, company=company, top_k=5, min_overlap=0.0)
            right = retriever.search(query, company)
            self.assertEqual(
                [doc.doc_id for doc, _score in left],
                [doc.doc_id for doc, _score in right],
                msg=query,
            )
            for (_doc, left_score), (_other, right_score) in zip(left, right):
                self.assertTrue(
                    math.isclose(left_score, right_score, rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE)
                )
        self.assertIn("generation", retriever.last_diagnostics)

        self.shards[2].close()
        self.shards = self.shards[:2]
        with self.assertRaises(RetrievalUnavailable):
            retriever.search("shared", None)

    def test_bad_request_does_not_stop_the_shard(self) -> None:
        client = self.clients[0]
        with self.assertRaises(Exception):
            client._request("POST", "/v1/search", {"query": 1})
        self.assertIn("ready", client.health())

    def test_fake_transport_failures(self) -> None:
        class Ready:
            def __init__(self, shard_id: str, generation: str = "gen"):
                self.shard_id = shard_id
                self.generation = generation

            def health(self):
                return {
                    "shard_id": self.shard_id,
                    "ready": True,
                    "generation": self.generation,
                    "document_count": 1,
                }

            def stats(self, terms):
                return {
                    "shard_id": self.shard_id,
                    "generation": self.generation,
                    "n_docs": 1,
                    "total_tokens": 1,
                    "df": {term: 0 for term in terms},
                }

            def search(self, payload):
                return {"generation": self.generation, "hits": []}

        mismatch = DistributedRetriever(
            [Ready("a", "one"), Ready("b", "two")],
            top_k=5,
            min_overlap=0.0,
            timeout=1.0,
            retries=0,
        )
        with self.assertRaises(RetrievalUnavailable):
            mismatch.search("alpha", None)

        class BadSearch(Ready):
            def search(self, payload):
                return {"generation": self.generation, "hits": "nope"}

        malformed = DistributedRetriever(
            [BadSearch("a"), BadSearch("b", "gen")],
            top_k=5,
            min_overlap=0.0,
            timeout=1.0,
            retries=0,
        )
        with self.assertRaises(RetrievalUnavailable):
            malformed.search("alpha", None)

        class Slow(Ready):
            def health(self):
                time.sleep(1.0)
                return super().health()

        slow = DistributedRetriever([Slow("a")], top_k=5, min_overlap=0.0, timeout=0.15, retries=0)
        with self.assertRaises(RetrievalUnavailable):
            slow.search("alpha", None)


if __name__ == "__main__":
    unittest.main()

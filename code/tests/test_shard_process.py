import csv
import socket
import sys
import tempfile
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cluster
from config.settings import DISTRIBUTED_SEARCH_MODE, RuntimeConfig
from data_loader.corpus_loader import load_corpus
from output.writer import FIELDNAMES
from pipeline.run_pipeline import run_pipeline
from retrieval.bm25 import BM25Index
from retrieval.distributed_retriever import DistributedRetriever, RetrievalUnavailable
from retrieval.shard_client import load_shard_clients, load_shard_members


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class ShardProcessTests(unittest.TestCase):
    def test_three_processes_match_bm25_and_fail_closed(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        pid_file = root / "pids.json"
        try:
            corpus = root / "corpus" / "hackerrank"
            corpus.mkdir(parents=True)
            (corpus / "handbook.md").write_text(
                "The alpha beta handbook explains the team workflow. "
                "Read the alpha beta handbook before you change a setting.\n",
                encoding="utf-8",
            )
            tickets = root / "tickets.csv"
            tickets.write_text(
                "Subject,Issue,Company\n"
                "Handbook,How do I use the alpha beta handbook for the team,HackerRank\n"
                "Movie,which movie should I watch tonight,HackerRank\n"
                "Theft,I am reporting identity theft on my account,HackerRank\n",
                encoding="utf-8",
            )
            config_path = root / "shards.json"
            shards = []
            for index in range(3):
                port = _free_port()
                shards.append(
                    {
                        "id": f"shard-{index}",
                        "url": f"http://127.0.0.1:{port}",
                        "db": str(root / f"shard-{index}.db"),
                    }
                )
            config_path.write_text(
                __import__("json").dumps({"shards": shards}),
                encoding="utf-8",
            )
            cluster.start_cluster(config_path, pid_file, root, timeout=15.0)
            report = cluster.index_cluster(config_path, root / "corpus", timeout=15.0)
            self.assertEqual(health_code := cluster.health_cluster(config_path, 5.0), 0)
            docs = load_corpus(root / "corpus")
            self.assertEqual(report["documents"], len(docs))
            reference = BM25Index(docs)
            clients = load_shard_clients(config_path, timeout=5.0)
            retriever = DistributedRetriever(clients, top_k=5, min_overlap=0.0, timeout=5.0, retries=0)
            expected = reference.search("alpha beta handbook", company=None, top_k=5, min_overlap=0.0)
            actual = retriever.search("alpha beta handbook", None)
            self.assertEqual(
                [doc.doc_id for doc, _score in expected],
                [doc.doc_id for doc, _score in actual],
            )

            output = root / "out.csv"
            log = root / "log.txt"
            runtime = RuntimeConfig(
                input_path=tickets,
                output_path=output,
                corpus_root=root / "corpus",
                sample_path=tickets,
                log_path=log,
                min_score=0.01,
                min_overlap=0.0,
                top_k=5,
                max_sentences=4,
                search_mode=DISTRIBUTED_SEARCH_MODE,
                shard_config_path=config_path,
                diagnostics_path=root / "diagnostics.jsonl",
                shard_timeout_seconds=5.0,
                shard_retries=0,
            )
            run_pipeline(runtime, _logger())
            rows = _read_csv(output)
            self.assertEqual(len(rows), 3)
            self.assertEqual(list(rows[0].keys()), FIELDNAMES)
            self.assertEqual(rows[0]["status"], "replied")
            self.assertIn("Evidence:", rows[0]["justification"])
            self.assertIn("Out-of-scope", rows[1]["justification"])
            self.assertNotIn("Distributed retrieval unavailable", rows[1]["justification"])
            self.assertEqual(rows[2]["status"], "escalated")
            self.assertIn("Escalation rule matched", rows[2]["justification"])
            self.assertTrue((root / "diagnostics.jsonl").exists())

            records = __import__("json").loads(pid_file.read_text(encoding="utf-8"))["processes"]
            cluster._stop_pid(int(records[0]["pid"]))
            with self.assertRaises(RetrievalUnavailable):
                retriever.search("alpha beta handbook", None)
            failed = root / "failed.csv"
            runtime_failed = RuntimeConfig(
                input_path=tickets,
                output_path=failed,
                corpus_root=root / "corpus",
                sample_path=tickets,
                log_path=log,
                min_score=0.01,
                min_overlap=0.0,
                top_k=5,
                max_sentences=4,
                search_mode=DISTRIBUTED_SEARCH_MODE,
                shard_config_path=config_path,
                diagnostics_path=None,
                shard_timeout_seconds=1.0,
                shard_retries=0,
            )
            run_pipeline(runtime_failed, _logger())
            failed_rows = _read_csv(failed)
            self.assertEqual(len(failed_rows), 3)
            self.assertEqual(list(failed_rows[0].keys()), FIELDNAMES)
            self.assertEqual(failed_rows[0]["status"], "escalated")
            self.assertIn("Distributed retrieval unavailable", failed_rows[0]["justification"])
            self.assertIn("Out-of-scope", failed_rows[1]["justification"])
            self.assertEqual(failed_rows[1]["status"], "replied")
            self.assertIn("Escalation rule matched", failed_rows[2]["justification"])
            self.assertNotIn("Distributed retrieval unavailable", failed_rows[2]["justification"])
        finally:
            cluster.stop_cluster(pid_file)
            tmp.cleanup()

    def test_replica_answers_after_the_primary_deadline(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        pid_file = root / "pids.json"
        deadline = 0.4
        try:
            corpus = root / "corpus" / "hackerrank"
            corpus.mkdir(parents=True)
            (corpus / "handbook.md").write_text(
                "The alpha beta handbook explains the team workflow.\n",
                encoding="utf-8",
            )
            shards = []
            for index in range(3):
                shards.append(
                    {
                        "id": f"shard-{index}",
                        "url": f"http://127.0.0.1:{_free_port()}",
                        "db": str(root / f"shard-{index}.db"),
                        "replica_url": f"http://127.0.0.1:{_free_port()}",
                        "replica_db": str(root / f"shard-{index}-replica.db"),
                    }
                )
            config_path = root / "shards.json"
            config_path.write_text(__import__("json").dumps({"shards": shards}), encoding="utf-8")
            cluster.start_cluster(config_path, pid_file, root, timeout=15.0)
            cluster.index_cluster(config_path, root / "corpus", timeout=15.0)
            docs = load_corpus(root / "corpus")
            reference = BM25Index(docs)
            primaries, replicas = load_shard_members(config_path, timeout=deadline)
            retriever = DistributedRetriever(
                primaries,
                top_k=5,
                min_overlap=0.0,
                timeout=deadline,
                retries=0,
                replicas=replicas,
            )
            expected = reference.search("alpha beta handbook", company=None, top_k=5, min_overlap=0.0)
            records = __import__("json").loads(pid_file.read_text(encoding="utf-8"))["processes"]
            primary = next(row for row in records if row["shard_id"] == "shard-0" and row["role"] == "primary")
            replica = next(row for row in records if row["shard_id"] == "shard-0" and row["role"] == "replica")
            cluster._stop_pid(int(primary["pid"]))
            started = __import__("time").perf_counter()
            actual = retriever.search("alpha beta handbook", None)
            elapsed = __import__("time").perf_counter() - started
            self.assertEqual(
                [doc.doc_id for doc, _score in expected],
                [doc.doc_id for doc, _score in actual],
            )
            self.assertGreaterEqual(elapsed, deadline * 2.5)
            cluster._stop_pid(int(replica["pid"]))
            with self.assertRaises(RetrievalUnavailable):
                retriever.search("alpha beta handbook", None)
        finally:
            cluster.stop_cluster(pid_file)
            tmp.cleanup()


def _logger():
    import logging

    logger = logging.getLogger("shard-process-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()

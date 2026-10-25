"""Measure full-scan BM25 against one, two, and three local shard processes.

From the repository root:

    python code/benchmarks/benchmark_distributed.py

The script starts its own processes on unused localhost ports and stops them
before exiting. ``--shard-config`` is accepted so the documented command works;
this run does not attach to or modify that cluster. Pass ``--output PATH`` to
write the JSON report. That file is a generated result.
"""

import argparse
import json
import math
import os
import platform
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import cluster
from benchmarks.benchmark_inverted_index import build_queries, percentile
from config.settings import MIN_OVERLAP, TOP_K, repo_root
from data_loader.corpus_loader import load_corpus
from retrieval.bm25 import BM25Index
from retrieval.distributed_retriever import DistributedRetriever, RetrievalUnavailable
from retrieval.shard_client import load_shard_clients


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _db_bytes(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix) if suffix else path
            if candidate.exists():
                total += candidate.stat().st_size
    return total


def _summarize(samples: Sequence[float]) -> Dict[str, float]:
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples) * 1000.0,
        "p50_ms": percentile(samples, 0.50) * 1000.0,
        "p95_ms": percentile(samples, 0.95) * 1000.0,
        "p99_ms": percentile(samples, 0.99) * 1000.0,
    }


def _measure_topology(corpus: Path, queries, reference, shard_count: int, repeats: int, root: Path) -> dict:
    config_path = root / f"shards-{shard_count}.json"
    pid_file = root / f"pids-{shard_count}.json"
    db_paths = []
    shards = []
    for index in range(shard_count):
        db_path = root / f"s{shard_count}-{index}.db"
        db_paths.append(db_path)
        shards.append(
            {
                "id": f"shard-{index}",
                "url": f"http://127.0.0.1:{_free_port()}",
                "db": str(db_path),
            }
        )
    config_path.write_text(json.dumps({"shards": shards}), encoding="utf-8")
    try:
        cluster.start_cluster(config_path, pid_file, root, timeout=20.0)
        started = time.perf_counter()
        report = cluster.index_cluster(config_path, corpus, timeout=120.0)
        build_seconds = time.perf_counter() - started
        clients = load_shard_clients(config_path, timeout=10.0)
        retriever = DistributedRetriever(
            clients,
            top_k=TOP_K,
            min_overlap=MIN_OVERLAP,
            timeout=10.0,
            retries=0,
        )
        mismatches = []
        for query in queries:
            left = reference.search(
                query["text"],
                company=query["company"],
                top_k=TOP_K,
                min_overlap=MIN_OVERLAP,
            )
            right = retriever.search(query["text"], query["company"])
            ids_match = [doc.doc_id for doc, _score in left] == [doc.doc_id for doc, _score in right]
            scores_match = all(
                math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
                for (_doc_a, a), (_doc_b, b) in zip(left, right)
            )
            if not ids_match or not scores_match:
                mismatches.append(query["text"][:80])
        for query in queries:
            retriever.search(query["text"], query["company"])
        samples = []
        for _ in range(repeats):
            for query in queries:
                started = time.perf_counter()
                retriever.search(query["text"], query["company"])
                samples.append(time.perf_counter() - started)
        unavailable = ""
        if shard_count > 1:
            records = json.loads(pid_file.read_text(encoding="utf-8"))["processes"]
            cluster._stop_pid(int(records[-1]["pid"]))
            try:
                retriever.search(queries[0]["text"], queries[0]["company"])
                unavailable = "search returned results after a shard was stopped"
            except RetrievalUnavailable as exc:
                unavailable = str(exc)
        return {
            "shard_count": shard_count,
            "documents": report["documents"],
            "build_seconds": build_seconds,
            "db_bytes": _db_bytes(db_paths),
            "parity_mismatches": mismatches,
            "warm_query_ms": _summarize(samples),
            "unavailable_shard": unavailable,
        }
    finally:
        cluster.stop_cluster(pid_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local full scan against shard processes")
    parser.add_argument(
        "--shard-config",
        default="",
        help="Accepted for the documented command. This benchmark uses isolated ports.",
    )
    parser.add_argument("--corpus", default=str(repo_root() / "data"))
    parser.add_argument("--tickets", default=str(repo_root() / "support_tickets" / "support_tickets.csv"))
    parser.add_argument("--query-repeats", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.shard_config:
        print(f"ignoring live cluster in {args.shard_config}; using isolated localhost ports")

    docs = load_corpus(Path(args.corpus))
    if not docs:
        print(f"No documents loaded from {args.corpus}")
        return 1
    reference = BM25Index(docs)
    queries = build_queries(docs, reference, Path(args.tickets))
    full_samples = []
    for query in queries:
        reference.search(query["text"], company=query["company"], top_k=TOP_K, min_overlap=MIN_OVERLAP)
    for _ in range(args.query_repeats):
        for query in queries:
            started = time.perf_counter()
            reference.search(query["text"], company=query["company"], top_k=TOP_K, min_overlap=MIN_OVERLAP)
            full_samples.append(time.perf_counter() - started)
    started = time.perf_counter()
    BM25Index(docs)
    full_build = time.perf_counter() - started

    tmp = tempfile.TemporaryDirectory()
    try:
        topologies = [
            _measure_topology(Path(args.corpus), queries, reference, count, args.query_repeats, Path(tmp.name))
            for count in (1, 2, 3)
        ]
    finally:
        tmp.cleanup()

    report = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "corpus_documents": len(docs),
        "query_count": len(queries),
        "top_k": TOP_K,
        "min_overlap": MIN_OVERLAP,
        "query_repeats": args.query_repeats,
        "full_scan": {
            "build_seconds": full_build,
            "warm_query_ms": _summarize(full_samples),
        },
        "topologies": topologies,
    }
    print(f"python={report['environment']['python']} platform={report['environment']['platform']}")
    print(f"cpu_count={report['environment']['cpu_count']} documents={len(docs)} queries={len(queries)}")
    full = report["full_scan"]["warm_query_ms"]
    print(
        f"full_scan build={full_build:.3f}s warm p50={full['p50_ms']:.3f} "
        f"p95={full['p95_ms']:.3f} p99={full['p99_ms']:.3f}"
    )
    for item in topologies:
        warm = item["warm_query_ms"]
        print(
            f"shards={item['shard_count']} build={item['build_seconds']:.3f}s "
            f"db_bytes={item['db_bytes']} mismatches={len(item['parity_mismatches'])} "
            f"p50={warm['p50_ms']:.3f} p95={warm['p95_ms']:.3f} p99={warm['p99_ms']:.3f} "
            f"unavailable={item['unavailable_shard'][:80]}"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {output}")
    mismatches = sum(len(item["parity_mismatches"]) for item in topologies)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())

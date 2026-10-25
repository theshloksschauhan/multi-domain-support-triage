"""Measure replica disk and the latency after one primary is killed.

From the repository root:

    python code/benchmarks/measure_replicas.py

The script starts three partitions with a replica each, on unused localhost
ports, and stops them before exiting. A killed primary is read only after
its deadline, so the reported latency includes that wait plus the replica call
on every search phase.
"""

import json
import socket
import sys
import tempfile
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import cluster
from config.settings import MIN_OVERLAP, TOP_K, repo_root
from retrieval.distributed_retriever import DistributedRetriever
from retrieval.shard_client import load_shard_members

DEADLINE = 1.0


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _db_bytes(paths) -> int:
    total = 0
    details = {}
    for path in paths:
        file_total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix) if suffix else path
            if candidate.exists():
                size = candidate.stat().st_size
                file_total += size
                details[candidate.name] = size
        total += file_total
    return total, details


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="replica-measure-"))
    pid_file = root / "pids.json"
    corpus = repo_root() / "data"
    db_paths = []
    shards = []
    for index in range(3):
        primary_db = root / f"shard-{index}.db"
        replica_db = root / f"shard-{index}-replica.db"
        db_paths.extend([primary_db, replica_db])
        shards.append(
            {
                "id": f"shard-{index}",
                "url": f"http://127.0.0.1:{_free_port()}",
                "db": str(primary_db),
                "replica_url": f"http://127.0.0.1:{_free_port()}",
                "replica_db": str(replica_db),
            }
        )
    config_path = root / "shards.json"
    config_path.write_text(json.dumps({"shards": shards}), encoding="utf-8")
    try:
        cluster.start_cluster(config_path, pid_file, root, timeout=30.0)
        started = time.perf_counter()
        report = cluster.index_cluster(config_path, corpus, timeout=180.0)
        build_seconds = time.perf_counter() - started
        total, details = _db_bytes(db_paths)
        primaries, replicas = load_shard_members(config_path, timeout=DEADLINE)
        retriever = DistributedRetriever(
            primaries,
            top_k=TOP_K,
            min_overlap=MIN_OVERLAP,
            timeout=DEADLINE,
            retries=0,
            replicas=replicas,
        )
        query = "how do I pause a subscription"
        healthy_started = time.perf_counter()
        retriever.search(query, "HackerRank")
        healthy_seconds = time.perf_counter() - healthy_started
        records = json.loads(pid_file.read_text(encoding="utf-8"))["processes"]
        primary = next(row for row in records if row["shard_id"] == "shard-0" and row["role"] == "primary")
        cluster._stop_pid(int(primary["pid"]))
        failed_started = time.perf_counter()
        retriever.search(query, "HackerRank")
        failed_seconds = time.perf_counter() - failed_started
        print(
            json.dumps(
                {
                    "documents": report["documents"],
                    "generation": report["generation"],
                    "build_seconds": round(build_seconds, 3),
                    "db_bytes": total,
                    "files": details,
                    "deadline_seconds": DEADLINE,
                    "healthy_query_seconds": round(healthy_seconds, 3),
                    "killed_primary_query_seconds": round(failed_seconds, 3),
                },
                indent=2,
            )
        )
        return 0
    finally:
        cluster.stop_cluster(pid_file)


if __name__ == "__main__":
    sys.exit(main())

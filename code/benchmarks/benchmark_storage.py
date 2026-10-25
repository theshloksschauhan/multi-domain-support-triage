"""Compare the term-frequency layout with normalized SQLite postings.

From the repository root:

    python code/benchmarks/benchmark_storage.py

The real corpus decides which layout is shipped. A second table uses 200,000
generated chunks and is not written under data/.
"""

import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.benchmark_inverted_index import build_queries, percentile
from config.settings import MIN_OVERLAP, TOP_K, repo_root
from data_loader.corpus_loader import Document, load_corpus
from retrieval.bm25 import BM25Index
from retrieval.posting_index import PostingIndex
from retrieval.shard_store import ShardStore


BUILD_LIMIT_SECONDS = 60.0
BYTE_LIMIT = 400 * 1000 * 1000
SYNTHETIC_DOCS = 200_000


def _db_bytes(path: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix) if suffix else path
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def _p95_ms(index, queries, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        for query in queries:
            started = time.perf_counter()
            index.search(
                query["text"],
                company=query["company"],
                top_k=TOP_K,
                min_overlap=MIN_OVERLAP,
            )
            samples.append((time.perf_counter() - started) * 1000.0)
    return percentile(samples, 0.95)


def _measure_term_freq(docs, queries, root: Path, repeats: int) -> dict:
    path = root / "term-freq.db"
    started = time.perf_counter()
    store = ShardStore(path)
    store.replace_partition(docs, "storage-benchmark")
    index = store.load_index()
    store.close()
    build_seconds = time.perf_counter() - started
    return {
        "layout": "term_freq",
        "build_seconds": round(build_seconds, 3),
        "db_bytes": _db_bytes(path),
        "query_p95_ms": round(_p95_ms(index, queries, repeats), 3),
    }


def _measure_postings(docs, queries, root: Path, repeats: int) -> dict:
    path = root / "postings.db"
    index = PostingIndex(path)
    started = time.perf_counter()
    index.build(docs)
    build_seconds = time.perf_counter() - started
    try:
        p95 = _p95_ms(index, queries, repeats)
    finally:
        index.close()
    return {
        "layout": "postings",
        "build_seconds": round(build_seconds, 3),
        "db_bytes": _db_bytes(path),
        "query_p95_ms": round(p95, 3),
    }


def _winner(term_freq: dict, postings: dict) -> str:
    postings_ok = (
        postings["build_seconds"] < BUILD_LIMIT_SECONDS and postings["db_bytes"] < BYTE_LIMIT
    )
    if postings_ok and postings["query_p95_ms"] < term_freq["query_p95_ms"]:
        return "postings"
    return "term_freq"


def _synthetic(count: int) -> list:
    docs = []
    for index in range(count):
        tokens = [f"t{index % 40}", f"t{(index // 7) % 40}", "shared", f"w{index % 5}"]
        docs.append(
            Document(
                doc_id=f"synthetic-{index:06d}",
                company="HackerRank" if index % 3 else "Visa",
                product_area="general",
                text=" ".join(tokens),
                tokens=tokens,
                term_freq=Counter(tokens),
                path=Path("synthetic.md"),
            )
        )
    return docs


def _synthetic_queries() -> list:
    return [
        {"text": "shared t0", "company": None},
        {"text": "shared t3 w1", "company": "Visa"},
        {"text": "t10 w4", "company": "HackerRank"},
        {"text": "shared shared", "company": None},
    ]


def main() -> int:
    root = repo_root()
    docs = load_corpus(root / "data")
    reference = BM25Index(docs)
    queries = build_queries(docs, reference, root / "support_tickets" / "support_tickets.csv")
    repeats = 3
    with tempfile.TemporaryDirectory() as tmp:
        real_root = Path(tmp) / "real"
        real_root.mkdir()
        real = {
            "term_freq": _measure_term_freq(docs, queries, real_root, repeats),
            "postings": _measure_postings(docs, queries, real_root, repeats),
        }
        real["winner"] = _winner(real["term_freq"], real["postings"])
        synthetic_docs = _synthetic(SYNTHETIC_DOCS)
        synthetic_root = Path(tmp) / "synthetic"
        synthetic_root.mkdir()
        synthetic_queries = _synthetic_queries()
        synthetic = {
            "documents": SYNTHETIC_DOCS,
            "term_freq": _measure_term_freq(synthetic_docs, synthetic_queries, synthetic_root, 1),
            "postings": _measure_postings(synthetic_docs, synthetic_queries, synthetic_root, 1),
        }
    report = {
        "build_limit_seconds": BUILD_LIMIT_SECONDS,
        "byte_limit": BYTE_LIMIT,
        "real_corpus": real,
        "synthetic_corpus": synthetic,
        "shipped_layout": real["winner"],
    }
    destination = root / "support_tickets" / "storage_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

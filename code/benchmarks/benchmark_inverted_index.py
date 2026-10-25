"""Compare full-scan BM25Index with InvertedIndex on the local corpus.

From the repository root:

    python code/benchmarks/benchmark_inverted_index.py

The script prints the measured build times, warm query latency, and result
parity. Pass ``--output PATH`` to also write a JSON record. That file is a
generated result and is not required to run the benchmark again.
"""

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.settings import MIN_OVERLAP, TOP_K, repo_root
from data_loader.corpus_loader import Document, load_corpus
from data_loader.ticket_loader import load_tickets
from retrieval.bm25 import BM25Index
from retrieval.inverted_index import InvertedIndex
from utils.text_cleaning import tokenize


def percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("no samples")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def time_builds(factory: Callable[[Sequence[Document]], object], docs: Sequence[Document], repeats: int) -> List[float]:
    samples: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        factory(docs)
        samples.append(time.perf_counter() - started)
    return samples


def candidate_count(index: InvertedIndex, query: str) -> int:
    found = set()
    for term in tokenize(query):
        found.update(index.postings.get(term, ()))
    return len(found)


def same_results(reference: BM25Index, inverted: InvertedIndex, query: str, company: Optional[str]) -> bool:
    left = reference.search(query, company=company, top_k=TOP_K, min_overlap=MIN_OVERLAP)
    right = inverted.search(query, company=company, top_k=TOP_K, min_overlap=MIN_OVERLAP)
    if [doc.doc_id for doc, _score in left] != [doc.doc_id for doc, _score in right]:
        return False
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for (_doc_a, a), (_doc_b, b) in zip(left, right))


def measure_pair(
    reference: BM25Index,
    inverted: InvertedIndex,
    queries: Sequence[dict],
    repeats: int,
) -> Dict[str, Dict[str, List[float]]]:
    """Time both indexes on the same query before moving to the next one.

    Separate passes made the later pass look faster as the CPU warmed up.
    Interleaving keeps the comparison on one sample stream.
    """
    samples: Dict[str, Dict[str, List[float]]] = {
        "overall": {"full_scan": [], "inverted": []},
    }
    for repeat in range(repeats):
        for query in queries:
            label = query["label"]
            bucket = samples.setdefault(label, {"full_scan": [], "inverted": []})
            order = (
                (("full_scan", reference), ("inverted", inverted))
                if repeat % 2 == 0
                else (("inverted", inverted), ("full_scan", reference))
            )
            for name, index in order:
                started = time.perf_counter()
                index.search(query["text"], company=query["company"], top_k=TOP_K, min_overlap=MIN_OVERLAP)
                elapsed = time.perf_counter() - started
                bucket[name].append(elapsed)
                samples["overall"][name].append(elapsed)
    return samples


def summarize(samples: Sequence[float]) -> Dict[str, float]:
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples) * 1000.0,
        "p50_ms": percentile(samples, 0.50) * 1000.0,
        "p95_ms": percentile(samples, 0.95) * 1000.0,
    }


def build_queries(docs: Sequence[Document], reference: BM25Index, tickets_path: Path) -> List[dict]:
    queries: List[dict] = []
    if tickets_path.exists():
        for ticket in load_tickets(tickets_path):
            queries.append(
                {
                    "label": "ticket",
                    "text": f"{ticket.subject} {ticket.issue}".strip(),
                    "company": ticket.company or None,
                }
            )
    rare = next(term for term, df in reference.doc_freq.items() if df == 1)
    common = max(reference.doc_freq, key=reference.doc_freq.get)
    repeated_source = queries[0]["text"] if queries else "assessment assessment"
    queries.extend(
        [
            {"label": "rare", "text": rare, "company": None},
            {"label": "common", "text": common, "company": None},
            {"label": "repeated", "text": f"{repeated_source} {repeated_source}", "company": None},
            {"label": "repeated", "text": f"{common} {common} {common}", "company": None},
            {"label": "nomatch", "text": "zzzznotaterm999 xyzzyplugh", "company": None},
            {"label": "curated", "text": "visa dispute charge", "company": "Visa"},
            {"label": "curated", "text": "claude privacy crawler", "company": "Claude"},
            {"label": "curated", "text": "hackerrank assessment candidate", "company": "HackerRank"},
        ]
    )
    if not docs:
        raise RuntimeError("corpus produced no documents")
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark full-scan BM25 against the inverted index")
    parser.add_argument("--corpus", default=str(repo_root() / "data"))
    parser.add_argument("--tickets", default=str(repo_root() / "support_tickets" / "support_tickets.csv"))
    parser.add_argument("--build-repeats", type=int, default=5)
    parser.add_argument("--query-repeats", type=int, default=20)
    parser.add_argument("--output", default="", help="Optional JSON results path")
    args = parser.parse_args()

    load_started = time.perf_counter()
    docs = load_corpus(Path(args.corpus))
    load_seconds = time.perf_counter() - load_started
    if not docs:
        print(f"No documents loaded from {args.corpus}")
        return 1

    queries = build_queries(docs, BM25Index(docs), Path(args.tickets))
    reference_builds = time_builds(BM25Index, docs, args.build_repeats)
    inverted_builds = time_builds(InvertedIndex, docs, args.build_repeats)
    reference = BM25Index(docs)
    inverted = InvertedIndex(docs)
    for query in queries:
        reference.search(query["text"], company=query["company"], top_k=TOP_K, min_overlap=MIN_OVERLAP)
        inverted.search(query["text"], company=query["company"], top_k=TOP_K, min_overlap=MIN_OVERLAP)

    mismatches = [
        query["text"][:80]
        for query in queries
        if not same_results(reference, inverted, query["text"], query["company"])
    ]
    by_label: Dict[str, List[dict]] = {}
    for query in queries:
        by_label.setdefault(query["label"], []).append(query)

    timed = measure_pair(reference, inverted, queries, args.query_repeats)
    latency: Dict[str, Dict[str, dict]] = {}
    for label, group in by_label.items():
        latency[label] = {
            "full_scan": summarize(timed[label]["full_scan"]),
            "inverted": summarize(timed[label]["inverted"]),
            "median_inverted_candidates": statistics.median(
                candidate_count(inverted, query["text"]) for query in group
            ),
        }
    overall = {
        "full_scan": summarize(timed["overall"]["full_scan"]),
        "inverted": summarize(timed["overall"]["inverted"]),
    }

    report = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "corpus_documents": len(docs),
        "vocabulary": len(reference.doc_freq),
        "postings_entries": sum(len(indexes) for indexes in inverted.postings.values()),
        "query_count": len(queries),
        "top_k": TOP_K,
        "min_overlap": MIN_OVERLAP,
        "k1": reference.k1,
        "b": reference.b,
        "build_repeats": args.build_repeats,
        "query_repeats": args.query_repeats,
        "corpus_load_seconds": load_seconds,
        "build_median_ms": {
            "full_scan": statistics.median(reference_builds) * 1000.0,
            "inverted": statistics.median(inverted_builds) * 1000.0,
        },
        "parity_mismatches": mismatches,
        "latency_ms": {"overall": overall, "by_label": latency},
    }

    print(f"python={report['environment']['python']} platform={report['environment']['platform']}")
    print(f"cpu_count={report['environment']['cpu_count']} processor={report['environment']['processor']}")
    print(
        f"documents={report['corpus_documents']} vocabulary={report['vocabulary']} "
        f"postings={report['postings_entries']} queries={report['query_count']}"
    )
    print(f"top_k={TOP_K} min_overlap={MIN_OVERLAP} k1={reference.k1} b={reference.b}")
    print(
        f"build median ms: full_scan={report['build_median_ms']['full_scan']:.2f} "
        f"inverted={report['build_median_ms']['inverted']:.2f} "
        f"(repeats={args.build_repeats}, corpus load {load_seconds:.2f}s excluded)"
    )
    print(f"parity mismatches: {len(mismatches)}")
    full = overall["full_scan"]
    inv = overall["inverted"]
    print(
        f"overall warm query ms: full_scan mean={full['mean_ms']:.3f} "
        f"p50={full['p50_ms']:.3f} p95={full['p95_ms']:.3f} | "
        f"inverted mean={inv['mean_ms']:.3f} p50={inv['p50_ms']:.3f} p95={inv['p95_ms']:.3f} "
        f"(samples={full['count']})"
    )
    for label, group in latency.items():
        full = group["full_scan"]
        inv = group["inverted"]
        print(
            f"{label}: candidates~{group['median_inverted_candidates']:.0f} "
            f"full_scan p50={full['p50_ms']:.3f} p95={full['p95_ms']:.3f} "
            f"inverted p50={inv['p50_ms']:.3f} p95={inv['p95_ms']:.3f}"
        )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {output_path}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())

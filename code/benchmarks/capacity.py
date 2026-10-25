"""Choose the fewest partitions whose measured p95 is at most 80ms.

This is the linear program: minimize partition count subject to
``p95(n) <= 80``. The feasible points are the measured topologies, so
enumeration is the exact solution. The 80ms limit is fixed before the run.

From the repository root, after the distributed benchmark has written a report:

    python code/benchmarks/capacity.py support_tickets/distributed_report.json
"""

import json
import sys
from pathlib import Path


LIMIT_MS = 80.0


def solve(measurements) -> dict:
    feasible = [row for row in measurements if float(row["p95_ms"]) <= LIMIT_MS]
    if not feasible:
        return {
            "limit_ms": LIMIT_MS,
            "feasible": False,
            "chosen_partitions": None,
            "objective": "minimize partition count",
            "constraint": f"p95 <= {LIMIT_MS:.0f}ms",
        }
    chosen = min(feasible, key=lambda row: int(row["shard_count"]))
    return {
        "limit_ms": LIMIT_MS,
        "feasible": True,
        "chosen_partitions": int(chosen["shard_count"]),
        "chosen_p95_ms": float(chosen["p95_ms"]),
        "objective": "minimize partition count",
        "constraint": f"p95 <= {LIMIT_MS:.0f}ms",
        "measurements": measurements,
    }


def measurements_from_report(payload: dict) -> list:
    rows = []
    for item in payload["topologies"]:
        rows.append(
            {
                "shard_count": item["shard_count"],
                "p95_ms": item["warm_query_ms"]["p95_ms"],
            }
        )
    return rows


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python code/benchmarks/capacity.py REPORT.json", file=sys.stderr)
        return 2
    payload = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    result = solve(measurements_from_report(payload))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

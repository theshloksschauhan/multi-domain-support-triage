import sys
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from benchmarks.capacity import solve


class CapacityTests(unittest.TestCase):
    def test_fewest_partitions_under_the_limit(self) -> None:
        result = solve(
            [
                {"shard_count": 1, "p95_ms": 90.0},
                {"shard_count": 2, "p95_ms": 70.0},
                {"shard_count": 3, "p95_ms": 40.0},
            ]
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["chosen_partitions"], 2)
        self.assertEqual(result["limit_ms"], 80.0)

    def test_infeasible_when_every_topology_is_slower(self) -> None:
        result = solve(
            [
                {"shard_count": 1, "p95_ms": 120.0},
                {"shard_count": 3, "p95_ms": 81.0},
            ]
        )
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["chosen_partitions"])


if __name__ == "__main__":
    unittest.main()

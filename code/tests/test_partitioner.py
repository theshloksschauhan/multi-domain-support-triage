import os
import subprocess
import sys
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from retrieval.partitioner import SCHEMA_VERSION, assign_shard, index_generation


class PartitionerTests(unittest.TestCase):
    def test_assignment_is_complete_and_stable(self) -> None:
        doc_ids = [f"doc-{i}" for i in range(50)] + [r"claude\a.md::chunk0", "claude/a.md::chunk0"]
        for shard_count in (1, 2, 3):
            first = [assign_shard(doc_id, shard_count) for doc_id in doc_ids]
            second = [assign_shard(doc_id, shard_count) for doc_id in doc_ids]
            self.assertEqual(first, second)
            self.assertTrue(all(0 <= shard < shard_count for shard in first))
        self.assertEqual({assign_shard(doc_id, 1) for doc_id in doc_ids}, {0})
        self.assertEqual(
            assign_shard(r"claude\a.md::chunk0", 3),
            assign_shard(r"claude\a.md::chunk0", 3),
        )

    def test_assignment_matches_another_process(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_CODE_ROOT)
        doc_id = r"claude\a.md::chunk0"
        script = (
            "import sys; from retrieval.partitioner import assign_shard; "
            "print(assign_shard(sys.argv[1], 3))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, doc_id],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_CODE_ROOT),
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(int(completed.stdout.strip()), assign_shard(doc_id, 3))

    def test_generation_tracks_topology_and_corpus(self) -> None:
        ids = ["b", "a"]
        counts = [2, 3]
        base = index_generation(ids, counts, ["shard-0", "shard-1"], SCHEMA_VERSION)
        self.assertEqual(base, index_generation(["a", "b"], [3, 2], ["shard-0", "shard-1"]))
        self.assertNotEqual(base, index_generation(ids, counts, ["shard-1", "shard-0"]))
        self.assertNotEqual(base, index_generation(ids, [2, 4], ["shard-0", "shard-1"]))
        self.assertNotEqual(base, index_generation(ids, counts, ["shard-0", "shard-1", "shard-2"]))

    def test_shard_count_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            assign_shard("doc", 0)


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from retrieval.protocol import ProtocolError, commit_request, search_request
from retrieval.shard_client import load_shard_config, require_loopback


class ProtocolTests(unittest.TestCase):
    def test_search_rejects_booleans_used_as_numbers(self) -> None:
        payload = {
            "query": "alpha",
            "generation": "gen",
            "top_k": True,
            "stats": {"n_docs": 1, "total_tokens": 1, "df": {"alpha": 1}},
        }
        with self.assertRaises(ProtocolError):
            search_request(payload)

    def test_commit_rejects_bool_and_empty_generation(self) -> None:
        with self.assertRaises(ProtocolError):
            commit_request({"generation": "gen", "expected_count": True})
        with self.assertRaises(ProtocolError):
            commit_request({"generation": "", "expected_count": 1})
        self.assertEqual(commit_request({"generation": "gen", "expected_count": 0}), ("gen", 0))

    def test_config_requires_unique_loopback_shards(self) -> None:
        self.assertEqual(require_loopback("127.0.0.1"), "127.0.0.1")
        with self.assertRaises(ValueError):
            require_loopback("0.0.0.0")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shards.json"
            path.write_text(
                json.dumps(
                    {
                        "shards": [
                            {"id": "shard-0", "url": "http://127.0.0.1:8100"},
                            {"id": "shard-0", "url": "http://127.0.0.1:8101"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_shard_config(path)
            path.write_text(
                json.dumps({"shards": [{"id": "shard-0", "url": "http://example.com:8100"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_shard_config(path)


if __name__ == "__main__":
    unittest.main()

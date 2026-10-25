"""Deterministic document placement for a fixed shard list.

Assignment uses SHA-256 of the exact document ID bytes. Python's built-in
``hash()`` is not used because it changes between processes. The shard count
is the configured list length. Reordering or resizing that list is a full
rebuild, not a rebalance.
"""

import hashlib
import json
from typing import Sequence


SCHEMA_VERSION = 1


def assign_shard(doc_id: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def index_generation(
    doc_ids: Sequence[str],
    token_counts: Sequence[int],
    shard_ids: Sequence[str],
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Hash the corpus and the configured shard order.

    ``shard_ids`` is the topology order used by ``assign_shard``. Sorting it
    would hide a reorder that moves documents between processes.
    """
    if len(doc_ids) != len(token_counts):
        raise ValueError("doc_ids and token_counts must have the same length")
    pairs = sorted(zip(doc_ids, token_counts), key=lambda item: item[0])
    payload = {
        "schema": schema_version,
        "shard_ids": list(shard_ids),
        "docs": [[doc_id, int(count)] for doc_id, count in pairs],
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

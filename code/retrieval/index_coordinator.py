"""Build one generation across a fixed shard list.

Each primary shard receives only the documents assigned to its position. A
replica, when configured, receives that same partition and is not an extra
slot in the hash. The active generation flips only when a member commits the
expected count. The coordinator then checks that the union of primary ids is
exactly the corpus.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from data_loader.corpus_loader import load_corpus
from retrieval.distributed_retriever import RetrievalUnavailable
from retrieval.partitioner import SCHEMA_VERSION, assign_shard, index_generation
from retrieval.protocol import chunked, document_to_payload
from retrieval.shard_client import ShardClient, ShardClientError


class IndexCoordinator:
    def __init__(
        self,
        clients: Sequence[ShardClient],
        batch_size: int = 400,
        replicas: Optional[Sequence[Optional[ShardClient]]] = None,
    ):
        if not clients:
            raise ValueError("at least one shard is required")
        self.clients = list(clients)
        if len({client.shard_id for client in self.clients}) != len(self.clients):
            raise ValueError("shard ids must be unique")
        if replicas is None:
            self.replicas = [None] * len(self.clients)
        else:
            self.replicas = list(replicas)
        if len(self.replicas) != len(self.clients):
            raise ValueError("replica list must align with primary shards")
        self.batch_size = batch_size

    def build(self, corpus_root: Path) -> Dict[str, object]:
        docs = load_corpus(corpus_root)
        groups: Dict[str, list] = {client.shard_id: [] for client in self.clients}
        for doc in docs:
            slot = assign_shard(doc.doc_id, len(self.clients))
            groups[self.clients[slot].shard_id].append(doc)
        generation = index_generation(
            [doc.doc_id for doc in docs],
            [len(doc.tokens) for doc in docs],
            [client.shard_id for client in self.clients],
            SCHEMA_VERSION,
        )
        started = time.perf_counter()
        try:
            for client, replica in zip(self.clients, self.replicas):
                assigned = groups[client.shard_id]
                self._write_member(client, assigned, generation)
                if replica is not None:
                    self._write_member(replica, assigned, generation)
        except (ShardClientError, OSError, TimeoutError) as exc:
            raise RetrievalUnavailable(f"index build failed: {exc}") from exc
        self.verify(docs, generation)
        return {
            "generation": generation,
            "documents": len(docs),
            "seconds": time.perf_counter() - started,
            "per_shard": {shard_id: len(items) for shard_id, items in groups.items()},
        }

    def _write_member(self, client: ShardClient, assigned: list, generation: str) -> None:
        client.begin_index()
        for batch in chunked(assigned, self.batch_size):
            client.index_batch([document_to_payload(doc) for doc in batch])
        client.commit(generation, len(assigned))

    def _ready_ids(self, client: ShardClient, generation: str) -> List[str]:
        health = client.health()
        if health.get("generation") != generation or not health.get("ready"):
            raise RetrievalUnavailable(
                f"{client.shard_id} is not ready at generation {generation}"
            )
        return client.document_ids()

    def verify(self, docs, generation: str) -> None:
        expected = {doc.doc_id for doc in docs}
        found: List[str] = []
        try:
            for client, replica in zip(self.clients, self.replicas):
                primary_ids = self._ready_ids(client, generation)
                found.extend(primary_ids)
                if replica is not None:
                    replica_ids = self._ready_ids(replica, generation)
                    if set(replica_ids) != set(primary_ids):
                        raise RetrievalUnavailable(
                            f"{client.shard_id} replica does not match the primary partition"
                        )
        except ShardClientError as exc:
            raise RetrievalUnavailable(f"index verification failed: {exc}") from exc
        if len(found) != len(set(found)) or set(found) != expected:
            raise RetrievalUnavailable("partition is incomplete or contains duplicate ids")

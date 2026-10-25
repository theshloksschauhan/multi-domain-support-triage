"""Two-phase search across a fixed set of shard processes.

Phase 1 sums document counts, token counts, and document frequencies.
Phase 2 asks every partition to score with those same statistics, then merges
by descending score and ascending document id. Each partition has one primary.
When a replica is configured, a primary that misses its deadline is read from
that replica once. A partition with no live copy fails the query. The caller
must not treat that as an empty result set.
"""

import logging
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from data_loader.corpus_loader import Document
from retrieval.inverted_index import GlobalBm25Stats
from retrieval.shard_client import ShardClient, ShardClientError
from utils.text_cleaning import tokenize


class RetrievalUnavailable(RuntimeError):
    """Distributed retrieval cannot return a complete ranking."""


class DistributedRetriever:
    def __init__(
        self,
        clients: Sequence[ShardClient],
        top_k: int,
        min_overlap: float,
        timeout: float = 5.0,
        retries: int = 1,
        replicas: Optional[Sequence[Optional[ShardClient]]] = None,
    ):
        if not clients:
            raise RetrievalUnavailable("distributed search has no shards configured")
        self.clients = list(clients)
        if len({client.shard_id for client in self.clients}) != len(self.clients):
            raise RetrievalUnavailable("distributed search shard ids must be unique")
        if replicas is None:
            self.replicas = [None] * len(self.clients)
        else:
            self.replicas = list(replicas)
        if len(self.replicas) != len(self.clients):
            raise RetrievalUnavailable("replica list must align with primary shards")
        self.top_k = top_k
        self.min_overlap = min_overlap
        self.timeout = timeout
        self.retries = max(0, retries)
        self.last_diagnostics: dict = {}
        self._logger = logging.getLogger("triage.retrieval")

    def check_ready(self) -> str:
        """Return the generation shared by every ready shard."""
        rows = self._fanout("health", lambda client: client.health())
        return self._agreed_generation(rows)

    def search(self, query: str, company: Optional[str]) -> List[Tuple[Document, float]]:
        query_id = uuid.uuid4().hex[:12]
        diagnostics: dict = {
            "query_id": query_id,
            "shards": [],
            "candidate_count": 0,
            "generation": "",
            "result_count": 0,
        }
        self.last_diagnostics = diagnostics
        query_tokens = tokenize(query)
        if not query_tokens:
            try:
                diagnostics["generation"] = self.check_ready()
            except RetrievalUnavailable:
                raise
            return []

        health_rows = self._fanout("health", lambda client: client.health())
        generation = self._agreed_generation(health_rows)
        diagnostics["generation"] = generation
        terms = list(dict.fromkeys(query_tokens))
        stat_rows = self._fanout("stats", lambda client: client.stats(terms))
        for _shard_id, _elapsed, row in stat_rows:
            if row.get("generation") != generation:
                raise RetrievalUnavailable("stats generation does not match the active generation")
        n_docs = sum(int(row["n_docs"]) for _shard_id, _elapsed, row in stat_rows)
        total_tokens = sum(int(row["total_tokens"]) for _shard_id, _elapsed, row in stat_rows)
        df: Counter[str] = Counter()
        for _shard_id, _elapsed, row in stat_rows:
            frequencies = row.get("df")
            if not isinstance(frequencies, dict):
                raise RetrievalUnavailable("shard stats response is missing document frequencies")
            for term, value in frequencies.items():
                df[str(term)] += int(value)
        payload = {
            "query": query,
            "company": company,
            "top_k": self.top_k,
            "min_overlap": self.min_overlap,
            "generation": generation,
            "stats": {
                "n_docs": n_docs,
                "total_tokens": total_tokens,
                "df": dict(df),
            },
        }
        search_rows = self._fanout("search", lambda client: client.search(payload))
        merged: List[Tuple[Document, float]] = []
        seen = set()
        for shard_id, elapsed, row in search_rows:
            if row.get("generation") != generation:
                raise RetrievalUnavailable("search generation does not match the active generation")
            hits = row.get("hits")
            if not isinstance(hits, list):
                raise RetrievalUnavailable(f"{shard_id} returned malformed search hits")
            diagnostics["shards"].append(
                {
                    "shard_id": shard_id,
                    "latency_ms": round(elapsed * 1000.0, 3),
                    "candidates": len(hits),
                }
            )
            for hit in hits:
                if not isinstance(hit, dict) or "doc_id" not in hit or "score" not in hit:
                    raise RetrievalUnavailable(f"{shard_id} returned a malformed hit")
                doc_id = str(hit["doc_id"])
                if doc_id in seen:
                    raise RetrievalUnavailable(f"duplicate document id {doc_id}")
                seen.add(doc_id)
                merged.append((_document_from_hit(hit), float(hit["score"])))
        merged.sort(key=lambda item: (-item[1], item[0].doc_id))
        limited = merged[: self.top_k]
        diagnostics["candidate_count"] = len(merged)
        diagnostics["result_count"] = len(limited)
        self._logger.info(
            "query_id=%s generation=%s candidates=%d results=%d",
            query_id,
            generation,
            len(merged),
            len(limited),
        )
        return limited

    def _agreed_generation(self, rows) -> str:
        if not rows:
            raise RetrievalUnavailable("no shard health responses")
        generations = set()
        for shard_id, _elapsed, row in rows:
            if not row.get("ready"):
                raise RetrievalUnavailable(f"{shard_id} is not ready")
            generation = row.get("generation")
            if not isinstance(generation, str) or not generation:
                raise RetrievalUnavailable(f"{shard_id} has no active generation")
            generations.add(generation)
        if len(generations) != 1:
            raise RetrievalUnavailable("shard index generations do not agree")
        return next(iter(generations))

    def _read_partition(self, operation: str, primary: ShardClient, replica: Optional[ShardClient], fn):
        """Read one partition from its primary, then from its replica after the deadline."""
        started = time.perf_counter()
        if replica is None:
            last_error: Optional[BaseException] = None
            for _try in range(self.retries + 1):
                try:
                    value = fn(primary)
                    return primary.shard_id, time.perf_counter() - started, value
                except RetrievalUnavailable:
                    raise
                except (ShardClientError, OSError, TimeoutError, ValueError, TypeError) as exc:
                    last_error = exc
            raise RetrievalUnavailable(f"{primary.shard_id} {operation} failed: {last_error}")
        try:
            value = fn(primary)
            return primary.shard_id, time.perf_counter() - started, value
        except RetrievalUnavailable:
            raise
        except (ShardClientError, OSError, TimeoutError, ValueError, TypeError) as exc:
            primary_error: BaseException = exc
        remaining = self.timeout - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        try:
            value = fn(replica)
            return primary.shard_id, time.perf_counter() - started, value
        except RetrievalUnavailable:
            raise
        except (ShardClientError, OSError, TimeoutError, ValueError, TypeError) as exc:
            raise RetrievalUnavailable(
                f"{primary.shard_id} {operation} failed: primary {primary_error}; replica {exc}"
            ) from exc

    def _fanout(self, operation: str, fn):
        def attempt(primary: ShardClient, replica: Optional[ShardClient]):
            return self._read_partition(operation, primary, replica, fn)

        pool = ThreadPoolExecutor(max_workers=len(self.clients))
        futures = [
            pool.submit(attempt, primary, replica)
            for primary, replica in zip(self.clients, self.replicas)
        ]
        try:
            # Each attempt uses self.timeout. The wait has to cover every attempt,
            # or a retry starts only after the caller has already given up.
            # A replica is read after the primary deadline, so that path needs
            # one extra timeout for the second member.
            budget = self.timeout * (self.retries + 1)
            if any(replica is not None for replica in self.replicas):
                budget = self.timeout * 2 + 0.5
            done, pending = wait(futures, timeout=budget)
            if pending:
                for future in pending:
                    future.cancel()
                raise RetrievalUnavailable(f"shard {operation} timed out")
            results = []
            for future in done:
                results.append(future.result())
            return results
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


def _document_from_hit(hit: dict) -> Document:
    from utils.text_cleaning import tokenize as tokenize_text

    text = str(hit.get("text", ""))
    tokens = tokenize_text(text)
    return Document(
        doc_id=str(hit["doc_id"]),
        company=str(hit.get("company", "")),
        product_area=str(hit.get("product_area", "")),
        text=text,
        tokens=tokens,
        term_freq=Counter(tokens),
        path=Path(str(hit.get("source", ""))),
    )

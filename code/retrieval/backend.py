"""Search-backend interface for ticket retrieval.

The triage pipeline calls ``search(query, company)`` and does not know whether
results came from the in-process BM25 index or a later distributed coordinator.
Classification, risk rules, response generation, and CSV writing stay unchanged.

``local`` is the existing full-scan BM25 search. ``distributed`` uses the
shard cluster and raises ``RetrievalUnavailable`` rather than using the local
index when that cluster is missing or not ready.
"""

from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

from config.settings import DEFAULT_SEARCH_MODE, DISTRIBUTED_SEARCH_MODE
from data_loader.corpus_loader import Document
from retrieval.bm25 import BM25Index
from retrieval.distributed_retriever import DistributedRetriever, RetrievalUnavailable
from retrieval.retriever import Retriever
from retrieval.shard_client import ShardClientError, load_shard_members


class SearchBackend(Protocol):
    def search(self, query: str, company: Optional[str]) -> List[Tuple[Document, float]]:
        """Return the highest-scoring documents for one ticket query.

        Order matches ``BM25Index.search``: descending score, then ascending
        ``doc_id``. ``company`` uses the same filter, including the ``"None"``
        sentinel that means "do not filter".
        """


class LocalSearchBackend:
    """In-process BM25 baseline. Delegates to the existing ``Retriever``."""

    def __init__(self, index: BM25Index, min_overlap: float, top_k: int) -> None:
        self.retriever = Retriever(index, min_overlap, top_k)

    def search(self, query: str, company: Optional[str]) -> List[Tuple[Document, float]]:
        return self.retriever.search(query, company)


def build_search_backend(
    mode: str,
    docs: Sequence[Document],
    min_overlap: float,
    top_k: int,
    shard_config: Optional[str] = None,
    timeout: float = 5.0,
    retries: int = 1,
) -> Optional[SearchBackend]:
    """Build the backend selected by ``mode``.

    An empty corpus in local mode returns ``None``, matching the previous
    pipeline behavior of skipping retrieval. Distributed mode never builds the
    local full-scan index and never substitutes it when the cluster is down.
    """
    if mode == DISTRIBUTED_SEARCH_MODE:
        if not shard_config:
            raise RetrievalUnavailable(
                "Distributed search requires a shard config and does not fall back to local search."
            )
        try:
            clients, replicas = load_shard_members(Path(shard_config), timeout=timeout)
            retriever = DistributedRetriever(
                clients,
                top_k=top_k,
                min_overlap=min_overlap,
                timeout=timeout,
                retries=retries,
                replicas=replicas,
            )
            retriever.check_ready()
        except RetrievalUnavailable:
            raise
        except (OSError, ValueError, ShardClientError) as exc:
            raise RetrievalUnavailable(
                "Distributed search is unavailable and will not fall back to local search: "
                f"{exc}"
            ) from exc
        return retriever
    if mode != DEFAULT_SEARCH_MODE:
        raise ValueError(
            f"Unknown search mode {mode!r}. "
            f"Expected {DEFAULT_SEARCH_MODE!r} or {DISTRIBUTED_SEARCH_MODE!r}."
        )
    if not docs:
        return None
    index = BM25Index(list(docs))
    return LocalSearchBackend(index, min_overlap, top_k)

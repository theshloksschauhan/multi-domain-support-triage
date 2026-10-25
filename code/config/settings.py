import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Challenge schema values (lowercase)
STATUS_REPLIED = "replied"
STATUS_ESCALATED = "escalated"

OUT_OF_SCOPE_RESPONSE = (
    "Thanks for writing in. This message looks outside the scope of what we can help "
    "with through automated support."
)

ESCALATE_RESPONSE = (
    "Thanks for reaching out. Your request needs a closer look from the support team "
    "with access to your account and organization details."
)

MIN_SCORE = 10.0
MIN_OVERLAP = 0.08
TOP_K = 5
MAX_SENTENCES = 4

# Search backends. "local" is the in-process BM25 baseline. "distributed" is
# selected through the same interface and must not silently run local search.
DEFAULT_SEARCH_MODE = "local"
DISTRIBUTED_SEARCH_MODE = "distributed"
SEARCH_MODES = (DEFAULT_SEARCH_MODE, DISTRIBUTED_SEARCH_MODE)
DEFAULT_SHARD_TIMEOUT = 5.0
DEFAULT_SHARD_RETRIES = 1
DEFAULT_BODY_LIMIT = 32 * 1024 * 1024

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_log_path() -> Path:
    """Keep logs beside inputs by default for reproducible submission artifacts."""
    return repo_root() / "support_tickets" / "log.txt"


@dataclass(frozen=True)
class RuntimeConfig:
    input_path: Path
    output_path: Path
    corpus_root: Path
    sample_path: Path
    log_path: Path
    min_score: float
    min_overlap: float
    top_k: int
    max_sentences: int
    search_mode: str = DEFAULT_SEARCH_MODE
    shard_config_path: Optional[Path] = None
    diagnostics_path: Optional[Path] = None
    shard_timeout_seconds: float = DEFAULT_SHARD_TIMEOUT
    shard_retries: int = DEFAULT_SHARD_RETRIES


def default_config() -> RuntimeConfig:
    root = repo_root()
    return RuntimeConfig(
        input_path=root / "support_tickets" / "support_tickets.csv",
        output_path=root / "support_tickets" / "output.csv",
        corpus_root=root / "data",
        sample_path=root / "support_tickets" / "sample_support_tickets.csv",
        log_path=default_log_path(),
        min_score=MIN_SCORE,
        min_overlap=MIN_OVERLAP,
        top_k=TOP_K,
        max_sentences=MAX_SENTENCES,
        search_mode=DEFAULT_SEARCH_MODE,
        shard_config_path=None,
        diagnostics_path=None,
        shard_timeout_seconds=DEFAULT_SHARD_TIMEOUT,
        shard_retries=DEFAULT_SHARD_RETRIES,
    )

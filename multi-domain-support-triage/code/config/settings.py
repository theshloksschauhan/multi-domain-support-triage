import os
from dataclasses import dataclass
from pathlib import Path


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
    )

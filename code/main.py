
import argparse
import logging
import sys
from pathlib import Path

from config.settings import (
    DEFAULT_SHARD_RETRIES,
    DEFAULT_SHARD_TIMEOUT,
    MAX_SENTENCES,
    MIN_OVERLAP,
    MIN_SCORE,
    SEARCH_MODES,
    TOP_K,
    RuntimeConfig,
    default_config,
)
from pipeline.run_pipeline import run_pipeline
from utils.logger import setup_logging


def build_arg_parser(defaults: RuntimeConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Support ticket triage agent")
    parser.add_argument(
        "--input",
        default=str(defaults.input_path),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default=str(defaults.output_path),
        help="Output CSV path",
    )
    parser.add_argument(
        "--corpus",
        default=str(defaults.corpus_root),
        help="Corpus root directory",
    )
    parser.add_argument(
        "--sample",
        default=str(defaults.sample_path),
        help="Sample CSV path for schema hints",
    )
    parser.add_argument(
        "--log",
        default=str(defaults.log_path),
        help="Log file path",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help="Minimum retrieval score required to reply",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=MIN_OVERLAP,
        help="Minimum query token overlap with a document chunk",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Maximum retrieved chunks to score",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=MAX_SENTENCES,
        help="Maximum sentences to include in a response",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODES,
        default=defaults.search_mode,
        help=(
            "Retrieval backend. 'local' keeps in-process BM25 (the default). "
            "'distributed' queries the shard cluster and does not fall back to local search."
        ),
    )
    parser.add_argument(
        "--shard-config",
        default="",
        help="JSON file listing shard ids and localhost URLs. Required for distributed mode.",
    )
    parser.add_argument(
        "--diagnostics",
        default="",
        help="Optional JSONL path for retrieval diagnostics. Omitted by default.",
    )
    parser.add_argument(
        "--shard-timeout",
        type=float,
        default=defaults.shard_timeout_seconds or DEFAULT_SHARD_TIMEOUT,
        help="Per-request timeout in seconds for each shard call.",
    )
    parser.add_argument(
        "--shard-retries",
        type=int,
        default=defaults.shard_retries if defaults.shard_retries is not None else DEFAULT_SHARD_RETRIES,
        help="Extra attempts for an idempotent shard read after the first failure.",
    )
    return parser


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        corpus_root=Path(args.corpus),
        sample_path=Path(args.sample),
        log_path=Path(args.log),
        min_score=args.min_score,
        min_overlap=args.min_overlap,
        top_k=args.top_k,
        max_sentences=args.max_sentences,
        search_mode=args.search_mode,
        shard_config_path=Path(args.shard_config) if args.shard_config else None,
        diagnostics_path=Path(args.diagnostics) if args.diagnostics else None,
        shard_timeout_seconds=args.shard_timeout,
        shard_retries=args.shard_retries,
    )


def main() -> int:
    defaults = default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args()
    config = build_runtime_config(args)
    setup_logging(config.log_path)
    logger = logging.getLogger("triage")
    try:
        run_pipeline(config, logger)
    except Exception:
        logger.exception("Fatal error during run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


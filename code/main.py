
import argparse
import logging
import sys
from pathlib import Path

from config.settings import (
    MAX_SENTENCES,
    MIN_OVERLAP,
    MIN_SCORE,
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


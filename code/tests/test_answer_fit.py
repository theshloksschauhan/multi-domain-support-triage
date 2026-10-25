import csv
import json
import logging
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from config.settings import default_config, repo_root
from output.writer import FIELDNAMES
from pipeline.run_pipeline import run_pipeline


class AnswerFitTests(unittest.TestCase):
    def test_labeled_tickets_match_the_corpus(self) -> None:
        expectations = json.loads(
            (Path(__file__).parent / "fixtures" / "ticket_expectations.json").read_text(encoding="utf-8")
        )
        root = repo_root()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output.csv"
            config = replace(
                default_config(),
                output_path=output,
                log_path=Path(tmp) / "log.txt",
            )
            logger = logging.getLogger("answer-fit-test")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            run_pipeline(config, logger)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with (root / "support_tickets" / "support_tickets.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                tickets = list(csv.DictReader(handle))

        self.assertEqual(len(rows), len(tickets))
        self.assertEqual(list(rows[0].keys()), FIELDNAMES)
        by_subject = {
            (ticket.get("Subject") or "").strip(): row
            for ticket, row in zip(tickets, rows)
        }
        for expected in expectations:
            row = by_subject[expected["subject"]]
            self.assertEqual(row["status"], expected["status"], expected["subject"])
            response = row["response"].lower()
            justification = row["justification"].lower()
            if "response_contains" in expected:
                self.assertIn(expected["response_contains"].lower(), response, expected["subject"])
            if "response_excludes" in expected:
                self.assertNotIn(expected["response_excludes"].lower(), response, expected["subject"])
            if "justification_contains" in expected:
                self.assertIn(
                    expected["justification_contains"].lower(),
                    justification,
                    expected["subject"],
                )


if __name__ == "__main__":
    unittest.main()

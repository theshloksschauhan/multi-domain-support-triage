import json
import sys
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from prediction.features import company_is_ambiguous
from prediction.model import decision_cost, load_model, train_model


class DecisionModelTests(unittest.TestCase):
    def test_company_tie_is_ambiguous_and_one_match_is_not(self) -> None:
        self.assertTrue(
            company_is_ambiguous("the visa card failed a hackerrank test", "Card failed", "")
        )
        self.assertFalse(company_is_ambiguous("How do I dispute a visa charge", "Dispute", ""))
        self.assertFalse(
            company_is_ambiguous("How do I dispute a visa charge", "Dispute", "Visa")
        )

    def test_training_is_deterministic_and_separates_a_clear_reply(self) -> None:
        replied = {
            "score": 40.0,
            "overlap": 0.8,
            "action_match": 1.0,
            "risk": 0.0,
            "company_ambiguous": 0.0,
        }
        escalated = {
            "score": 4.0,
            "overlap": 0.05,
            "action_match": 0.0,
            "risk": 0.0,
            "company_ambiguous": 0.0,
        }
        rows = [replied, escalated] * 8
        labels = ["replied", "escalated"] * 8
        first = train_model(rows, labels, steps=200)
        second = train_model(rows, labels, steps=200)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.predict_status(replied), "replied")
        self.assertEqual(first.predict_status(escalated), "escalated")
        self.assertEqual(decision_cost("escalated", "replied"), 5)
        self.assertEqual(decision_cost("replied", "escalated"), 1)

    def test_saved_model_matches_the_held_out_report(self) -> None:
        model = load_model()
        report = json.loads(
            (_CODE_ROOT.parent / "support_tickets" / "decision_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(model.threshold, report["threshold"])
        self.assertEqual(report["tickets"], 29)
        self.assertIn("reply_precision", report)
        self.assertIn("escalation_precision", report)
        self.assertIn("two ordinary words", report["note"])


if __name__ == "__main__":
    unittest.main()

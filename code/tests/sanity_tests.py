import io
import sys
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from classification.request_type import classify_request_type
from classification.risk_detection import risk_escalate
from classification.product_area import infer_company
from routing.decision_engine import decide_status


class SanityTests(unittest.TestCase):
    def test_request_types(self) -> None:
        self.assertEqual(
            classify_request_type("Feature request for dark mode", ""),
            "feature_request",
        )
        self.assertEqual(classify_request_type("App is down", ""), "bug")
        self.assertEqual(classify_request_type("Which actor won", ""), "invalid")

    def test_decide_status(self) -> None:
        self.assertEqual(decide_status("invalid", "", [], 10.0), "replied")
        self.assertEqual(decide_status("product_issue", "", [], 10.0), "escalated")

    def test_risk_paraphrase_escalates_and_howto_does_not(self) -> None:
        escalate, reason = risk_escalate("Someone stole my identity last night", "")
        self.assertTrue(escalate)
        self.assertIn("Sensitive", reason)
        escalate, reason = risk_escalate("How do I reset my password", "Account help")
        self.assertFalse(escalate)
        self.assertEqual(reason, "")

    def test_conflicting_company_keywords_do_not_pick_the_first(self) -> None:
        self.assertEqual(
            infer_company("The visa card failed a hackerrank test", "", ""),
            "None",
        )
        self.assertEqual(infer_company("How do I dispute a visa charge", "", ""), "Visa")


def run() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SanityTests)
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run()
    print("sanity tests passed")

"""Write the frozen decision labels.

The original 29 tickets are the test split. Their labels are the rule
decisions. Later tickets use the same rules, except the known off-topic
requests are marked escalated even when those rules reply.

From the repository root:

    python code/prediction/build_labels.py
"""

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from classification.product_area import infer_company
from classification.request_type import classify_request_type
from classification.risk_detection import risk_escalate
from config.settings import MAX_SENTENCES, MIN_OVERLAP, MIN_SCORE, TOP_K, repo_root
from data_loader.corpus_loader import load_corpus
from data_loader.ticket_loader import load_tickets
from prediction.rules import rule_status
from retrieval.bm25 import BM25Index


OFF_TOPIC_SUBJECTS = {
    "Weather in the US Virgin Islands",
    "Prime numbers script",
    "Cricket score from last night",
    "Translate a grocery list",
    "College laptop recommendation",
    "Plan a Lisbon trip",
    "Explain photosynthesis",
    "Pantry pasta recipe",
    "Current time in Tokyo",
    "Novel recommendation",
    "Ounces in a liter",
    "Birthday message for a manager",
}


def label_path() -> Path:
    return CODE_ROOT / "tests" / "fixtures" / "decision_labels.json"


def build() -> list:
    root = repo_root()
    docs = load_corpus(root / "data")
    index = BM25Index(docs)
    original = {
        (ticket.subject.strip(), ticket.issue.strip())
        for ticket in load_tickets(root / "support_tickets" / "support_tickets.csv")
    }
    rows = []
    for ticket in load_tickets(root / "support_tickets" / "expanded_support_tickets.csv"):
        company = infer_company(ticket.issue, ticket.subject, ticket.company)
        request_type = classify_request_type(ticket.issue, ticket.subject)
        query = f"{ticket.subject} {ticket.issue}".strip()
        _risk_hit, risk_reason = risk_escalate(ticket.issue, ticket.subject)
        results = index.search(query, company=company, top_k=TOP_K, min_overlap=MIN_OVERLAP) if query else []
        top_doc = results[0][0] if results else None
        top_score = results[0][1] if results else 0.0
        status, _reason, _mismatch = rule_status(
            request_type,
            risk_reason,
            company,
            top_score,
            top_doc,
            query,
            MIN_SCORE,
            MAX_SENTENCES,
        )
        subject = ticket.subject.strip()
        split = "test" if (subject, ticket.issue.strip()) in original else "train"
        if split == "train" and subject in OFF_TOPIC_SUBJECTS:
            status = "escalated"
        rows.append(
            {
                "subject": subject,
                "issue": ticket.issue.strip(),
                "split": split,
                "status": status,
            }
        )
    return rows


def main() -> int:
    rows = build()
    destination = label_path()
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    test = sum(1 for row in rows if row["split"] == "test")
    train = sum(1 for row in rows if row["split"] == "train")
    print(f"wrote {len(rows)} labels ({test} test, {train} train) to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

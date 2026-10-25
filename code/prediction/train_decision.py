"""Train the decision model on the expanded tickets and score the frozen 29.

From the repository root:

    python code/prediction/train_decision.py
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
from prediction.build_labels import label_path
from prediction.features import model_features, ticket_features
from prediction.model import load_model, precision_report, save_model, train_model
from retrieval.bm25 import BM25Index


def _lookup(labels):
    return {(row["subject"], row["issue"]): row for row in labels}


def _decide(model, request_type, risk_hit, features):
    if risk_hit:
        return "escalated"
    if request_type == "invalid":
        return "replied"
    return model.predict_status(features)


def main() -> int:
    root = repo_root()
    labels = json.loads(label_path().read_text(encoding="utf-8"))
    by_key = _lookup(labels)
    docs = load_corpus(root / "data")
    index = BM25Index(docs)
    tickets = load_tickets(root / "support_tickets" / "expanded_support_tickets.csv")
    train_rows = []
    train_labels = []
    test_labels = []
    test_predicted = []
    for ticket in tickets:
        key = (ticket.subject.strip(), ticket.issue.strip())
        label = by_key[key]
        company = infer_company(ticket.issue, ticket.subject, ticket.company)
        request_type = classify_request_type(ticket.issue, ticket.subject)
        query = f"{ticket.subject} {ticket.issue}".strip()
        risk_hit, _reason = risk_escalate(ticket.issue, ticket.subject)
        results = index.search(query, company=company, top_k=TOP_K, min_overlap=MIN_OVERLAP) if query else []
        top_doc = results[0][0] if results else None
        top_score = results[0][1] if results else 0.0
        features = ticket_features(
            ticket.issue,
            ticket.subject,
            ticket.company,
            query,
            top_doc,
            top_score,
            MAX_SENTENCES,
        )
        seen = model_features(features, MIN_SCORE)
        if label["split"] == "train" and request_type != "invalid" and not risk_hit:
            train_rows.append(seen)
            train_labels.append(label["status"])
        if label["split"] == "test":
            test_labels.append(label["status"])
            test_predicted.append(None)  # filled after the model exists
            # stash features for the second pass via a side list
            test_predicted[-1] = (request_type, risk_hit, seen, label["status"])

    model = train_model(train_rows, train_labels)
    save_model(model)
    held_out_labels = []
    held_out_predicted = []
    for request_type, risk_hit, features, status in test_predicted:
        held_out_labels.append(status)
        held_out_predicted.append(_decide(model, request_type, risk_hit, features))
    report = precision_report(held_out_labels, held_out_predicted)
    report["train_rows"] = len(train_rows)
    report["threshold"] = model.threshold
    report["weights"] = dict(zip(
        ["score", "overlap", "action_match", "risk", "company_ambiguous"],
        model.weights,
    ))
    destination = root / "support_tickets" / "decision_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {destination}")
    # Touch load_model so a bad file fails in this process.
    load_model()
    return 0


if __name__ == "__main__":
    sys.exit(main())

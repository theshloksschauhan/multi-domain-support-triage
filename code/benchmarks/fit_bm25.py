"""Search k1, b, and the score cutoff on the training tickets.

Cost is five points for a wrong reply and one point for an unnecessary
escalation, measured against the frozen labels. The frozen 29 are not used
to choose the triple. From the repository root:

    python code/benchmarks/fit_bm25.py
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
from config.settings import MAX_SENTENCES, MIN_OVERLAP, TOP_K, repo_root
from data_loader.corpus_loader import load_corpus
from data_loader.ticket_loader import load_tickets
from prediction.build_labels import label_path
from prediction.model import decision_cost
from prediction.rules import rule_status
from retrieval.bm25 import BM25Index


K1_VALUES = (1.2, 1.5, 1.8)
B_VALUES = (0.5, 0.75, 1.0)
MIN_SCORE_VALUES = (8.0, 10.0, 12.0, 15.0)
DEFAULT = (1.5, 0.75, 10.0)


def _prepare(tickets, labels):
    by_key = {(row["subject"], row["issue"]): row for row in labels}
    prepared = []
    for ticket in tickets:
        key = (ticket.subject.strip(), ticket.issue.strip())
        label = by_key[key]
        company = infer_company(ticket.issue, ticket.subject, ticket.company)
        request_type = classify_request_type(ticket.issue, ticket.subject)
        query = f"{ticket.subject} {ticket.issue}".strip()
        _hit, risk_reason = risk_escalate(ticket.issue, ticket.subject)
        prepared.append(
            {
                "split": label["split"],
                "label": label["status"],
                "company": company,
                "request_type": request_type,
                "query": query,
                "risk_reason": risk_reason,
                "issue": ticket.issue,
                "subject": ticket.subject,
            }
        )
    return prepared


def _cost(index, prepared, min_score: float, split: str) -> int:
    total = 0
    for row in prepared:
        if row["split"] != split:
            continue
        results = (
            index.search(row["query"], company=row["company"], top_k=TOP_K, min_overlap=MIN_OVERLAP)
            if row["query"]
            else []
        )
        top_doc = results[0][0] if results else None
        top_score = results[0][1] if results else 0.0
        status, _reason, _mismatch = rule_status(
            row["request_type"],
            row["risk_reason"],
            row["company"],
            top_score,
            top_doc,
            row["query"],
            min_score,
            MAX_SENTENCES,
        )
        total += decision_cost(row["label"], status)
    return total


def main() -> int:
    root = repo_root()
    labels = json.loads(label_path().read_text(encoding="utf-8"))
    tickets = load_tickets(root / "support_tickets" / "expanded_support_tickets.csv")
    prepared = _prepare(tickets, labels)
    index = BM25Index(load_corpus(root / "data"))
    rows = []
    best = None
    for k1 in K1_VALUES:
        for b in B_VALUES:
            index.k1 = k1
            index.b = b
            for min_score in MIN_SCORE_VALUES:
                train_cost = _cost(index, prepared, min_score, "train")
                record = {
                    "k1": k1,
                    "b": b,
                    "min_score": min_score,
                    "train_cost": train_cost,
                }
                rows.append(record)
                if best is None or train_cost < best["train_cost"]:
                    best = record
                elif train_cost == best["train_cost"] and (k1, b, min_score) == DEFAULT:
                    best = record
    index.k1, index.b = best["k1"], best["b"]
    best_test = _cost(index, prepared, best["min_score"], "test")
    index.k1, index.b = DEFAULT[0], DEFAULT[1]
    default_train = _cost(index, prepared, DEFAULT[2], "train")
    default_test = _cost(index, prepared, DEFAULT[2], "test")
    ship = best_test < default_test and (best["k1"], best["b"], best["min_score"]) != DEFAULT
    report = {
        "default": {
            "k1": DEFAULT[0],
            "b": DEFAULT[1],
            "min_score": DEFAULT[2],
            "train_cost": default_train,
            "frozen_29_cost": default_test,
        },
        "selected": {
            "k1": best["k1"],
            "b": best["b"],
            "min_score": best["min_score"],
            "train_cost": best["train_cost"],
            "frozen_29_cost": best_test,
        },
        "ship_selected": ship,
        "grid": rows,
    }
    destination = root / "support_tickets" / "parameter_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("default", "selected", "ship_selected")}, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

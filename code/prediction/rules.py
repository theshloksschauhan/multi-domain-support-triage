"""The rule decision used to label tickets and to score parameter fits.

This is the pipeline's behavior before the logistic model: risk and outage
escalate, an invalid ticket is refused, a modest score with no company
escalates, and a reply whose sentences miss the requested action escalates.
"""

from typing import Optional, Tuple

from classification.answer_fit import reply_addresses_request
from data_loader.corpus_loader import Document
from generation.response_builder import select_snippets


def rule_status(
    request_type: str,
    risk_reason: str,
    company: str,
    top_score: float,
    top_doc: Optional[Document],
    query: str,
    min_score: float,
    max_sentences: int,
) -> Tuple[str, str, bool]:
    """Return status, the risk text used for that status, and answer mismatch."""

    reason = risk_reason
    if not reason and company == "None" and 0 < top_score < 15.0:
        reason = "Ambiguous organization field with modest retrieval confidence"
    if reason:
        return "escalated", reason, False
    if request_type == "invalid":
        return "replied", reason, False
    if top_doc is None or top_score < min_score:
        return "escalated", reason, False
    snippets = select_snippets(top_doc.text, query, max_sentences)
    if not reply_addresses_request(query, snippets):
        return "escalated", reason, True
    return "replied", reason, False

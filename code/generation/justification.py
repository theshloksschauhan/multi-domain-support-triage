"""Short, audit-friendly justification strings."""

from __future__ import annotations


def build_justification(
    status: str,
    request_type: str,
    risk_reason: str,
    top_score: float,
    min_score: float,
    has_docs: bool,
    retrieval_unavailable: bool = False,
    evidence_ref: str = "",
    answer_mismatch: bool = False,
) -> str:
    if retrieval_unavailable and status == "escalated":
        text = (
            "Distributed retrieval unavailable. The ticket was escalated "
            "instead of answered from incomplete evidence."
        )
    elif answer_mismatch and status == "escalated":
        text = "Retrieved document does not address the requested action."
    elif status == "escalated" and risk_reason:
        text = f"Escalation rule matched ({risk_reason.strip('.')})."
    elif status == "escalated" and not has_docs:
        text = (
            f"No adequate documentation match (best retrieval score {top_score:.1f}; "
            f"threshold {min_score:.1f})."
        )
    elif status == "escalated":
        text = (
            f"Retrieval confidence below threshold (best score {top_score:.1f}; "
            f"minimum {min_score:.1f})."
        )
    elif request_type == "invalid":
        text = "Out-of-scope ticket; sent a safe refusal without document claims."
    else:
        text = (
            f"Low-risk issue with strong documentation match (score {top_score:.1f}) "
            f"above threshold {min_score:.1f}."
        )
    if evidence_ref and status == "replied" and request_type != "invalid":
        return f"{text} Evidence: {evidence_ref}."
    return text

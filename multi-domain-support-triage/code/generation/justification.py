"""Short, audit-friendly justification strings."""

from __future__ import annotations


def build_justification(
    status: str,
    request_type: str,
    risk_reason: str,
    top_score: float,
    min_score: float,
    has_docs: bool,
) -> str:
    if status == "escalated" and risk_reason:
        return f"Escalation rule matched ({risk_reason.strip('.')})."

    if status == "escalated" and not has_docs:
        return (
            f"No adequate documentation match (best retrieval score {top_score:.1f}; "
            f"threshold {min_score:.1f})."
        )

    if status == "escalated":
        return (
            f"Retrieval confidence below threshold (best score {top_score:.1f}; "
            f"minimum {min_score:.1f})."
        )

    if request_type == "invalid":
        return "Out-of-scope ticket; sent a safe refusal without document claims."

    return (
        f"Low-risk issue with strong documentation match (score {top_score:.1f}) "
        f"above threshold {min_score:.1f}."
    )

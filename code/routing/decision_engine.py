from typing import List, Tuple

from data_loader.corpus_loader import Document


def decide_status(
    request_type: str,
    risk_reason: str,
    results: List[Tuple[Document, float]],
    min_score: float,
) -> str:
    if risk_reason:
        return "escalated"
    if request_type == "invalid":
        return "replied"
    if not results:
        return "escalated"
    top_score = results[0][1]
    if top_score < min_score:
        return "escalated"
    return "replied"

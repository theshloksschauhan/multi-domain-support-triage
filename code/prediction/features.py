"""Features the decision model is allowed to see.

The five values are the top BM25 score, token overlap, whether the selected
sentences contain the requested action, whether a risk or outage phrase
matched, and whether an empty company field names more than one organization.
"""

from typing import Dict, Optional

from classification.answer_fit import reply_addresses_request
from classification.product_area import COMPANY_KEYWORDS, normalize_company
from classification.risk_detection import risk_escalate
from data_loader.corpus_loader import Document
from generation.response_builder import select_snippets
from utils.text_cleaning import overlap_ratio, tokenize


FEATURE_NAMES = (
    "score",
    "overlap",
    "action_match",
    "risk",
    "company_ambiguous",
)


def company_is_ambiguous(issue: str, subject: str, company: str) -> bool:
    """True when the company cell is empty and at least two organizations match."""

    if normalize_company(company) != "None":
        return False
    text = f"{subject} {issue}".lower()
    matches = [
        name
        for name, keywords in COMPANY_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    return len(matches) > 1


def model_features(features: Dict[str, float], min_score: float) -> Dict[str, float]:
    """Return a copy whose score is divided by the run's score cutoff.

    Training uses the default cutoff of 10. A one-document index produces a
    small absolute score, and dividing by the cutoff configured for that run
    keeps the value on the training scale.
    """

    scaled = dict(features)
    cutoff = float(min_score) if min_score and min_score > 0 else 1.0
    scaled["score"] = float(features["score"]) / cutoff
    return scaled


def ticket_features(
    issue: str,
    subject: str,
    company: str,
    query: str,
    top_doc: Optional[Document],
    top_score: float,
    max_sentences: int,
) -> Dict[str, float]:
    overlap = 0.0
    action_match = 0.0
    if top_doc is not None and query:
        overlap = overlap_ratio(tokenize(query), top_doc.term_freq.keys())
        snippets = select_snippets(top_doc.text, query, max_sentences)
        if reply_addresses_request(query, snippets):
            action_match = 1.0
    risk_hit, _reason = risk_escalate(issue, subject)
    return {
        "score": float(top_score),
        "overlap": float(overlap),
        "action_match": action_match,
        "risk": 1.0 if risk_hit else 0.0,
        "company_ambiguous": 1.0 if company_is_ambiguous(issue, subject, company) else 0.0,
    }

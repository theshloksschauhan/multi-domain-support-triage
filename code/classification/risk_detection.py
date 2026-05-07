"""Rule-based risk detection for conservative escalation (stdlib only)."""

from __future__ import annotations

import re
from typing import Pattern, Tuple


# Multi-word phrases: substring match on normalized text
HIGH_RISK_PHRASES: Tuple[str, ...] = (
    "identity theft",
    "identity has been stolen",
    "identity stolen",
    "stolen identity",
    "unauthorized access",
    "security vulnerability",
    "bug bounty",
    "reveal internal",
    "show internal",
    "internal logic",
    "internal rules",
    "delete all files",
    "wipe system",
    "destroy data",
    "ignore previous",
    "disregard previous",
    "restore my access",
    "regain access",
    "removed my seat",
    "workspace owner",
    "not the workspace owner",
    "increase my score",
    "graded me unfairly",
    "tell the company to",
    "move me to the next round",
    "review my answers",
    "ban the seller",
    "wrong product",
    "merchant sent",
    "merchant is ignoring",
    "order id",
    "cs_live",
    "cs_test",
    "filling in the forms",
    "infosec process",
    "incorrect on the certificate",
    "name is incorrect",
    "update my name on the certificate",
    "dispute a charge with",
    "how do i dispute a charge on my account",
    "urgent cash",
    "cash advance",
    "payroll advance",
    "reveal the exact",
    "internal documents",
    "logique exacte",
    "règles internes",
    "règles internes",
)

# Single tokens / short words: whole-word match (avoids "fraud" matching inside "fraude")
HIGH_RISK_WORD_PATTERNS: Tuple[Pattern[str], ...] = tuple(
    re.compile(rf"\b{w}\b", re.IGNORECASE)
    for w in (
        "fraud",
        "fraudulent",
        "hacked",
        "vulnerability",
        "exploit",
    )
)

OUTAGE_PHRASES: Tuple[str, ...] = (
    "site is down",
    "none of the pages",
    "all requests are failing",
    "stopped working completely",
    "resume builder is down",
    "none of the submissions",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _phrase_hit(normalized: str, phrase: str) -> bool:
    return phrase in normalized


def risk_escalate(issue: str, subject: str) -> Tuple[bool, str]:
    """
    Return (escalate, reason). Empty reason means low risk for these rules.
    """
    normalized = _normalize(f"{subject} {issue}")

    for phrase in HIGH_RISK_PHRASES:
        if _phrase_hit(normalized, phrase):
            return True, "Sensitive, adversarial, account-specific, or policy-heavy request"

    for pat in HIGH_RISK_WORD_PATTERNS:
        if pat.search(normalized):
            return True, "Sensitive, adversarial, account-specific, or policy-heavy request"

    for phrase in OUTAGE_PHRASES:
        if _phrase_hit(normalized, phrase):
            return True, "Widespread outage or critical platform failure reported"

    # Broad outage wording
    if "down" in normalized and ("site" in normalized or "service" in normalized):
        return True, "Widespread outage or critical platform failure reported"

    return False, ""

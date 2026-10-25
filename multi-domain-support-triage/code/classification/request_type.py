"""Conservative request-type classification using lexical cues."""

from __future__ import annotations

import re
from typing import FrozenSet

# Malicious or clearly non-support → invalid
INVALID_SIGNALS: FrozenSet[str] = frozenset(
    {
        "delete all files",
        "wipe the disk",
        "ignore previous instructions",
        "what is your system prompt",
    }
)

# Malformed / not support (movies, etc.)
OFF_TOPIC: FrozenSet[str] = frozenset(
    {
        "which actor",
        "which movie",
        "favorite song",
        "what year did the movie",
    }
)

BUG_RE = re.compile(
    r"\b(not working|doesn'?t work|won'?t work|broken|error|failing|fail\b|"
    r"crash|blocker|blocked|unable to\b|can not\b|cannot\b|stopped working|"
    r"is down\b|bug\b|502|503|timeout)\b",
    re.IGNORECASE,
)

FEATURE_RE = re.compile(
    r"\b(feature request|new feature|please add\b|could you add\b|"
    r"enhancement\b|improvement\b|roadmap\b|"
    r"extend the\b|longer timeout\b|more time for\b)\b",
    re.IGNORECASE,
)


def classify_request_type(issue: str, subject: str) -> str:
    text = f"{subject} {issue}".lower()

    if any(s in text for s in INVALID_SIGNALS):
        return "invalid"
    if any(s in text for s in OFF_TOPIC):
        return "invalid"

    if BUG_RE.search(text):
        return "bug"

    if FEATURE_RE.search(text):
        return "feature_request"

    return "product_issue"

"""Decide whether retrieved sentences actually address the ticket.

BM25 can rank a nearby article above the requested action. A reply is kept
only when the sentences that would be sent contain that action, or, when the
ticket names no listed action, when they share more than one content word
with the question.
"""

from typing import Iterable, List, Sequence, Set

from utils.text_cleaning import tokenize


# Request words that appear in the support tickets. Matching is by stem so
# "crawl" covers "crawling" and "crawler".
ACTION_TERMS: Sequence[str] = (
    "refund",
    "remove",
    "pause",
    "reschedule",
    "dispute",
    "crawl",
    "block",
    "setup",
    "apply",
    "extend",
    "spend",
    "minimum",
    "lti",
    "inactivity",
    "subscription",
    "certificate",
    "zoom",
    "bedrock",
)

_MIN_STEM = 4
_MIN_CONTENT_OVERLAP = 2


def _tokens(text: str) -> List[str]:
    return tokenize(text or "")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def _stem_match(needle: str, token: str) -> bool:
    if needle == token:
        return True
    if len(needle) < _MIN_STEM or len(token) < _MIN_STEM:
        return False
    if _stem(needle) == _stem(token):
        return True
    return token.startswith(needle) or needle.startswith(token)


def action_terms(query: str) -> Set[str]:
    found = set()
    for token in _tokens(query):
        for action in ACTION_TERMS:
            if _stem_match(action, token):
                found.add(action)
    return found


def reply_addresses_request(query: str, snippets: Iterable[str]) -> bool:
    """Return whether ``snippets`` are specific enough to send as the reply."""

    snippet_text = " ".join(snippet for snippet in snippets if snippet)
    snippet_tokens = set(_tokens(snippet_text))
    if not snippet_tokens:
        return False
    requested = action_terms(query)
    if requested:
        return any(
            _stem_match(action, token)
            for action in requested
            for token in snippet_tokens
        )
    query_tokens = set(_tokens(query))
    return len(query_tokens & snippet_tokens) >= _MIN_CONTENT_OVERLAP

from pathlib import Path
from typing import Dict, List

from utils.text_cleaning import normalize_area


COMPANY_KEYWORDS: Dict[str, List[str]] = {
    "HackerRank": ["hackerrank", "assessment", "test", "candidate", "recruiter", "interview"],
    "Claude": ["claude", "anthropic", "bedrock", "lti"],
    "Visa": ["visa", "card", "merchant", "charge", "cheque", "travel", "atm"],
}


def normalize_company(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned or cleaned.lower() == "none":
        return "None"
    for name in COMPANY_KEYWORDS:
        if cleaned.lower() == name.lower():
            return name
    return cleaned


def infer_company(issue: str, subject: str, company: str) -> str:
    normalized = normalize_company(company)
    if normalized != "None":
        return normalized
    text = f"{subject} {issue}".lower()
    for name, keywords in COMPANY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return name
    return "None"


def infer_company_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "hackerrank" in parts:
        return "HackerRank"
    if "claude" in parts:
        return "Claude"
    if "visa" in parts:
        return "Visa"
    return "None"


def infer_product_area_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "hackerrank_community" in parts:
        return "community"
    if "privacy-and-legal" in parts:
        return "privacy"
    if "travel-support" in path.stem or "travel" in parts:
        return "travel_support"
    if "cheque" in path.stem:
        return "travel_support"
    if "general-help" in parts:
        return "general_help"
    for i, part in enumerate(parts):
        if part in {"hackerrank", "claude", "visa"} and i + 1 < len(parts):
            return normalize_area(parts[i + 1]) or "general"
    return "general"


def infer_product_area_from_ticket(issue: str, subject: str, company: str) -> str:
    text = f"{subject} {issue}".lower()
    if company == "Visa":
        if "travel" in text or "cheque" in text:
            return "travel_support"
        if "card" in text or "charge" in text or "merchant" in text:
            return "general_support"
        return "general_support"
    if company == "Claude":
        if "privacy" in text or "data" in text:
            return "privacy"
        if "conversation" in text or "chat" in text:
            return "conversation_management"
        return "general"
    if company == "HackerRank":
        if "community" in text:
            return "community"
        if "test" in text or "assessment" in text or "candidate" in text:
            return "screen"
        return "general"
    return "general"

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from utils.helpers import first_value


@dataclass(frozen=True)
class Ticket:
    issue: str
    subject: str
    company: str


def load_tickets(input_path: Path) -> List[Ticket]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")
    tickets: List[Ticket] = []
    with input_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            issue = first_value(row, ["Issue", "issue"]).strip()
            subject = first_value(row, ["Subject", "subject"]).strip()
            company = first_value(row, ["Company", "company"]).strip()
            tickets.append(Ticket(issue=issue, subject=subject, company=company))
    return tickets

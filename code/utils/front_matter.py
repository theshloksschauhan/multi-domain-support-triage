"""Minimal YAML front matter helpers (stdlib only)."""

from __future__ import annotations

import re
from typing import List


def extract_breadcrumbs(fm: str) -> List[str]:
    """Parse breadcrumb titles from a YAML front matter block."""
    if not fm or "breadcrumbs:" not in fm:
        return []
    lines = fm.splitlines()
    items: List[str] = []
    collecting = False
    for line in lines:
        if line.strip().startswith("breadcrumbs:"):
            collecting = True
            continue
        if not collecting:
            continue
        m = re.match(r'^\s*-\s+"(.*)"\s*$', line) or re.match(r"^\s*-\s+'(.*)'\s*$", line)
        if m:
            items.append(m.group(1))
            continue
        m2 = re.match(r"^\s*-\s+(.+)$", line)
        if m2:
            raw = m2.group(1).strip().strip('"').strip("'")
            items.append(raw)
            continue
        # Next YAML key at root level of front matter (not a list continuation)
        stripped = line.strip()
        if stripped and not stripped.startswith("-") and ":" in stripped and not stripped.startswith("#"):
            break
    return items

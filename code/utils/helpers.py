from typing import Dict, Iterable


def first_value(row: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return ""

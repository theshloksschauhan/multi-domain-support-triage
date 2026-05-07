import csv
from pathlib import Path
from typing import Dict, List


FIELDNAMES = ["status", "product_area", "response", "justification", "request_type"]


def write_output(output_path: Path, rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            clean = {k: (row.get(k) or "").replace("\r\n", "\n").replace("\r", "\n") for k in FIELDNAMES}
            writer.writerow(clean)

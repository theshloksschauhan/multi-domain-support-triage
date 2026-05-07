# Multi-domain support triage agent

Deterministic, corpus-grounded ticket triage for the HackerRank Orchestrate challenge. The agent indexes local Markdown support articles with BM25, applies conservative risk rules, classifies request types, and either replies using extracted sentences or escalates.

## Layout

- `config/settings.py` — paths, thresholds (`MIN_SCORE`, `MIN_OVERLAP`, `TOP_K`), deterministic defaults.
- `data_loader/` — CSV tickets and corpus chunks (with YAML breadcrumbs when present).
- `retrieval/` — BM25 index + overlap-filtered retrieval; company-aware filtering when `Company` is set.
- `classification/` — request type, risk phrases, product-area fallbacks.
- `routing/decision_engine.py` — replied vs escalated from risk + retrieval scores.
- `generation/` — snippet extraction and short justifications.
- `output/writer.py` — CSV with columns  
  `status,product_area,response,justification,request_type` (challenge schema).
- `pipeline/run_pipeline.py` — end-to-end orchestration and structured logging.

## Dependencies

Python 3.10+ standard library only (`requirements.txt` documents this).

## Run

From the **repository root** (parent of `code/` and `data/`):

```bash
python code/main.py \
  --input support_tickets/support_tickets.csv \
  --output support_tickets/output.csv \
  --corpus data \
  --log support_tickets/log.txt
```

Optional flags: `--min-score`, `--min-overlap`, `--top-k`, `--max-sentences`.

The script directory is on `sys.path`, so imports resolve without extra `PYTHONPATH`.

## Outputs

- **Predictions CSV** — exactly five columns, lowercase `status` (`replied` / `escalated`), allowed `request_type` values.
- **log.txt** — per-ticket retrieval score, optional risk reason, top chunk id, issue preview.

## Submission packaging

1. Zip **only** the `code/` directory (exclude `data/`, `support_tickets/`, virtualenvs, caches).
2. Attach generated `output.csv` and `log.txt` as required by the challenge.

## Assumptions

- Corpus lives under `data/` as `.md` files; metadata is taken from paths and front matter `breadcrumbs` when available.
- Escalation is preferred when risk rules fire or BM25 score is below `MIN_SCORE`.
- No network calls at runtime.

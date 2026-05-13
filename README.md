# Multi-domain Support Triage Agent

## Overview

This repository implements an advanced, terminal-based AI agent that automatically triages real-world support tickets spanning three distinct product ecosystems: **HackerRank**, **Claude**, and **Visa**. The agent is architected to operate exclusively using the **local support documentation corpus** provided in the repository, strictly avoiding any external network requests or non-approved sources for both security and reproducibility.

The project is written 100% in Python, leverages only the Python standard library, and is fully deterministic for fair comparison and auditing. Its core purpose is to automate and standardize the support ticket resolution process, maximize correctness, minimize risk (by escalating ambiguous or sensitive cases), and provide human-readable justifications that ensure trust and transparency.

---

## Objectives

- **Automated Triage:** Classify and respond to support tickets using local product documentation only.
- **Multi-domain Capability:** Support multiple product domains, adapting retrieval and classification based on product context.
- **Complete Reproducibility:** No internet connection or external APIs; corpus and tickets are fully local.
- **Transparency and Explainability:** Every decision (reply, escalation, routing) is accompanied by an explicit, corpus-grounded justification.
- **Safe Handling:** Escalate tickets with insufficient documentation coverage, ambiguous cases, or high risk.

---

## Solution Architecture

### 1. Input/Output Schema

#### Input: Support Tickets (CSV)

Each ticket includes columns such as ticket ID, product, subject, and message text. See `support_tickets/support_tickets.csv` for schema.

#### Output: Predictions (CSV)

| Column         | Description                                                   | Allowed Values                                       |
|----------------|--------------------------------------------------------------|------------------------------------------------------|
| status         | Action taken                                                 | `replied`, `escalated`                               |
| product_area   | Most relevant documentation category/domain area              | (free-form, based on corpus content)                 |
| response       | User-facing answer, grounded in documentation                 | (generated text, only facts/steps from the corpus)   |
| justification  | Concise explanation for the decision, based on evidence      | (generated text, must reference support content)     |
| request_type   | Type of user request                                         | `product_issue`, `feature_request`, `bug`, `invalid` |

---

### 2. File and Directory Structure

```
.
├── code/
│   ├── main.py                     # Pipeline entry point—run the whole process
│   ├── config/
│   │   └── settings.py             # Global settings for scoring and processing
│   ├── data_loader/                # Ticket loader, corpus chunker, and YAML parser
│   ├── retrieval/                  # BM25 search, overlap filtering, company/domain logic
│   ├── classification/             # Request/intent classification, risk rules, product mapping
│   ├── routing/decision_engine.py  # Escalate/reply logic
│   ├── generation/                 # Text generation modules for response and justification
│   └── output/writer.py            # Output CSV writer, schema validation
├── data/
│   ├── hackerrank/                 # Local HackerRank documentation
│   ├── claude/                     # Local Claude documentation
│   └── visa/                       # Local Visa documentation
├── support_tickets/
│   ├── sample_support_tickets.csv  # Example tickets for offline testing
│   ├── support_tickets.csv         # Main tickets to triage
│   └── output.csv                  # Your model's outputs
└── README.md                       # This file
```

---

### 3. How It Works: Step-by-Step

#### a. Data Loading
- Corpus articles are loaded from the local `data/` directory.
- Support tickets are loaded from CSV.
- Articles are parsed, chunked into retrieval-friendly units, and optionally annotated with breadcrumbs and YAML metadata for more granular search results.

#### b. Retrieval (BM25-based)
- For each incoming ticket, a **BM25** search is performed over all corpus chunks to score relevance.
- Top-K relevant chunks are further filtered based on direct keyword overlap and, if specified, product/company tagging.

#### c. Risk & Request-Type Classification
- The ticket is analyzed for keywords or patterns indicating bugs, feature requests, product issues, or invalid/unsupported cases.
- High-risk tickets (sensitive language, unsupported issues, legal or payment topics, insufficient evidence) are escalated automatically, avoiding hallucinations or unsupported advice.

#### d. Routing Decision
- Rules combine retrieval score, risk identification, product context, and evidence overlap to determine whether the agent can reply confidently or must escalate.

#### e. Response and Justification Generation
- For replied tickets, the most relevant corpus snippets are synthesized into a user response, strictly sourcing wording and steps from the documentation.
- A concise justification is attached, mapping evidence and decision process in natural language, explaining why a certain answer or escalation was chosen.

#### f. Output Generation
- A single row is written per ticket to `support_tickets/output.csv` with all the required columns.
- Detailed per-ticket logs (retrieval scores, debug info, reasons for escalation, etc.) are written to `support_tickets/log.txt`.

---

### 4. Determinism and Reproducibility

- All random choices (for sampling or tie-breaking) are seeded.
- The output is fully reproducible given the same corpus, settings, and input files.
- No external dependencies: the approach relies only on the Python standard library and runs fully offline.

---

### 5. Usage Instructions

#### Requirements

- Python 3.10 or newer (no external libraries required)

#### Running the Agent

From the root repository directory (parent of `code/`):

```bash
python code/main.py \
  --input support_tickets/support_tickets.csv \
  --output support_tickets/output.csv \
  --corpus data \
  --log support_tickets/log.txt
```

**Optional flags:**

- `--min-score`: Minimum retrieval score for a valid reply (default defined in `config/settings.py`)
- `--min-overlap`: Minimum keyword overlap required in candidate docs
- `--top-k`: How many top chunks to retrieve and consider
- `--max-sentences`: Maximum sentences to use in a generated answer

#### Output

1. `support_tickets/output.csv` — predictions (see schema above).
2. `support_tickets/log.txt` — logs for inspection/validation.

---

### 6. Detailed Module Descriptions

- **`code/config/settings.py`**  
  Centralized configuration for retrieval scoring, overlap, top-K, and random seed for deterministic runs.

- **`code/data_loader/`**  
  - *Ticket Loader*: Reads and validates CSV input.
  - *Corpus Chunker*: Splits documentation into smaller units for retrieval.
  - *YAML Parser*: Extracts front matter and breadcrumbs for hierarchical relevance.

- **`code/retrieval/`**  
  Implements a BM25 algorithm for scoring text similarity, including relevance filtering and, when available, company/product-aware scoping.

- **`code/classification/`**  
  Handles classification for:
    - Request type (using rule-based phrase matching and fallback logic)
    - Risk detection (patterns for escalation)
    - Product-area mapping

- **`code/routing/decision_engine.py`**  
  Orchestrates the logic that turns retrieval scores, risk flags, and request type predictions into a final status: reply or escalation. Conservative by design.

- **`code/generation/`**  
  Supports generating clear, factually grounded answers and short justifications that map input → evidence → output.

- **`code/output/writer.py`**  
  Handles writing output in strict schema, ensures column order, and validates generated content to minimize formatting errors.

---

### 7. Tuning and Customization

- All thresholds, retrieval hyperparameters, and deterministic seeds are adjustable in `config/settings.py` or via command-line flags.
- To support a new domain, simply deposit Markdown/MDX/HTML documentation in a new subdirectory under `data/`, following the convention and ensuring file naming is logical.

---

### 8. Assumptions and Limitations

- **No external lookups:** All answers must be justified from the supplied corpus.
- **Escalation is prioritized** if there is ambiguity or informational gaps.
- **No deep learning or LLMs**: The agent uses efficient classical IR, robust rule sets, and structured heuristics.
- **No hardcoded credentials and no sensitive information checked into source control.**
- **Corpus files must be present and readable; empty/missing corpus will result in escalations for all tickets.**
- **Request types and allowed values are enforced rigidly to enable fair evaluation.**

---

### 9. Contributions and Extensions

For improvements, bug fixes, or proposals (agent enhancements, new retrieval models, etc.):

1. Fork the repository and create a feature branch.
2. Follow the code structure and adhere to deterministic and corpus-grounding constraints.
3. Submit a pull request with a description of your approach, changes, and any new validation tests or tickets.

---

## Summary

**multi-domain-support-triage** is a robust, transparent, and fully local Python system designed to automate multi-domain support ticket processing in mission-critical environments. Every prediction is justified and traceable, and the system is easy to read, extend, and review.

If you encounter any ambiguity or would like more technical details, please see the in-line comments throughout the codebase or open an issue/discussion!

---

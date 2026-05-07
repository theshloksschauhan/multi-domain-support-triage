# Multi-Domain Support Triage — AI Judge Interview Handbook

> **Purpose:** A complete, project-specific guide for the HackerRank Orchestrate **Multi-Domain Support Triage Challenge** interview.  
> **Your system:** Deterministic **BM25** retrieval → rule-based **risk** gates → **reply or escalate** → grounded snippets — **no external LLM at runtime**.

---

## 1. 🔥 Project Overview (Simple but Powerful)

### Beginner explanation (non-technical)

Imagine a **library assistant** who is only allowed to quote from approved books on the shelf. When someone asks a question, the assistant:

1. **Finds** the most relevant pages (search).
2. **Checks** whether the question is dangerous, legally sensitive, or needs a human (safety rules).
3. Either **reads aloud short sentences from those pages** or says: **“I need to hand this to a specialist.”**

There is **no imagination** step — only lookup + rules. That is exactly what your agent does with **support articles** instead of books.

### Technical explanation (interview-ready)

Your pipeline is a **lexical retrieval + policy layer**:

| Piece | Role |
|--------|------|
| **Corpus** | Local `.md` support docs under `data/`, chunked after YAML-aware parsing |
| **BM25** | Scores query tokens vs document chunks (probabilistic relevance, deterministic implementation) |
| **Classification** | Regex / keyword rules → `request_type` (bug, feature_request, product_issue, invalid) |
| **Risk rules** | Phrases + word-boundary patterns → escalate sensitive / adversarial / account-specific cases |
| **Decision** | If risk OR weak retrieval OR empty hits → **escalate**; else **reply** from snippets |
| **Output** | CSV + structured log (scores, doc ids, risk reasons) |

**No network calls, no LLM API** — same inputs always yield the same outputs (**deterministic**).

### Killer one-liner

> **“We built a deterministic, corpus-grounded triage agent: BM25 finds evidence, explicit rules decide when we’re allowed to speak from that evidence — everything else escalates to humans.”**

---

## 2. 🧠 Architecture Deep Dive

### Components (mapped to your codebase)

| Component | Responsibility | Typical location |
|-----------|----------------|------------------|
| **Corpus loader** | Walk `data/**/*.md`, strip front matter **safely** (YAML can contain `---` inside URLs), chunk body text, attach **company** + **product_area** (from YAML `breadcrumbs` when present) | `data_loader/corpus_loader.py`, `utils/text_cleaning.py`, `utils/front_matter.py` |
| **Chunking** | Sentence windows for retrieval granularity (overlap avoids cutting answers in half) | `utils/text_cleaning.py` (`chunk_text`) |
| **BM25 retrieval** | Score chunks; optional **company filter** so Visa tickets don’t pull Claude docs | `retrieval/bm25.py`, `retrieval/retriever.py` |
| **Request classification** | Conservative rules for `request_type` | `classification/request_type.py` |
| **Risk detection** | Escalation triggers (fraud, outage language, account manipulation, jailbreak-ish asks, etc.) | `classification/risk_detection.py` |
| **Escalation / decision** | Combine risk + score threshold → `replied` vs `escalated` | `routing/decision_engine.py`, `pipeline/run_pipeline.py` |
| **Response generation** | Sentence selection from **top chunk** only; optional HTML entity cleanup | `generation/response_builder.py` |
| **Output writer** | Strict CSV schema + newline safety | `output/writer.py` |
| **CLI / orchestration** | End-to-end run, logging | `main.py`, `pipeline/run_pipeline.py` |

### Pipeline (conceptual order)

```text
CSV ticket
    → normalize query (subject + issue)
    → infer company (from column + text cues)
    → BM25 search (optional company filter, min token overlap, top-k)
    → pick product_area (from best chunk / breadcrumbs / fallback)
    → classify request_type
    → run risk rules (+ optional “ambiguous company / weak score” rule)
    → decide_status(replied | escalated)
    → if replied: extract snippets; else: templated escalate message
    → write row + log trace
```

### Text-based flow diagram

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Load corpus │ ──► │ Build BM25   │ ──► │ Load tickets│
│  (chunks)   │     │    index     │     │   from CSV  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Write CSV   │ ◄── │ Reply vs     │ ◄── │ Per ticket: │
│  + log.txt  │     │  escalate    │     │ retrieve +  │
└─────────────┘     └──────────────┘     │ classify +  │
                           ▲              │ risk gate   │
                           │              └─────────────┘
                    ┌──────┴───────┐
                    │ Risk hit OR  │
                    │ score < thr  │
                    │ OR no chunks │
                    └──────────────┘
```

---

## 3. ⚙️ End-to-End Flow (CRITICAL)

### Exactly one ticket’s journey

<details>
<summary><strong>Expand: ticket lifecycle (step-by-step)</strong></summary>

1. **Input**  
   - Read row from `support_tickets/support_tickets.csv`: `Issue`, `Subject`, `Company`.

2. **Company inference**  
   - Use explicit `Company` when valid; else infer from keywords (HackerRank / Claude / Visa) in subject+issue.

3. **Query string**  
   - `query = subject + " " + issue` (this is what BM25 searches).

4. **Retrieval**  
   - Tokenize query (stopwords removed).  
   - For each chunk: skip if **company filter** excludes it (when company is known).  
   - Skip if **query–document token overlap** ratio &lt; `MIN_OVERLAP` (reduces irrelevant hits).  
   - BM25 score; sort by score descending, break ties by stable `doc_id`; keep **top-k**.

5. **Scores**  
   - Record **top_score** (best BM25 value). If no results, `top_score = 0`.

6. **Product area**  
   - Prefer **breadcrumb trail** from the top chunk (e.g. `Team and Enterprise plans · Billing`).  
   - Optionally prefix **company name** when it clarifies domain.

7. **Request type**  
   - Run conservative regex / keyword logic → `bug` | `feature_request` | `product_issue` | `invalid`.

8. **Risk detection**  
   - Match **multi-word phrases** (substring) for policies like outage wording, “restore access”, “order id”, etc.  
   - Match **single risky words** with **word boundaries** (so English `fraud` does not false-trigger inside French `fraude`).  
   - Optional extra rule: **`Company` missing / “None”** and **modest BM25 score** → treat as **ambiguous context** and escalate.

9. **Decision**  
   - If **risk_reason** non-empty → **escalated**.  
   - Else if **no retrieval results** → **escalated**.  
   - Else if **top_score &lt; MIN_SCORE** → **escalated**.  
   - Else if **request_type == invalid** (benign off-topic) → often **replied** with safe refusal text (per your policy).  
   - Else → **replied**.

10. **Response**  
    - **Replied:** select sentences from **top chunk** that overlap query tokens; join into one concise answer; **HTML-unescape** for readable text.  
    - **Escalated:** fixed professional template (“needs human review with account context”) — **no fabricated policy**.

11. **Justification**  
    - Short audit string: escalation reason **or** retrieval score vs threshold.

12. **Output**  
    - Append row: `status, product_area, response, justification, request_type`.

13. **Logging**  
    - Log index, status, company, risk (or `-`), score, top `doc_id`, issue preview.

</details>

---

## 4. 🧪 Example Walkthrough (VERY IMPORTANT)

### Synthetic example aligned with your agent’s behavior

**Input ticket**

| Field | Value |
|--------|--------|
| Subject | `Claude access lost` |
| Issue | `I lost access to my Claude team workspace after our IT admin removed my seat. Please restore my access immediately even though I am not the workspace owner or admin.` |
| Company | `Claude` |

**Step 1 — Retrieval**  
- Query contains strong product terms (`Claude`, `team`, `seat`, `admin`).  
- BM25 likely returns **high-scoring** chunks from **Team / Enterprise / admin** articles.

**Step 2 — Classification**  
- Looks like a **product/process** issue, not a bare bug → likely **`product_issue`**.

**Step 3 — Risk**  
- Phrases such as **“restore my access”**, **workspace/seat/admin ownership** match your **account-sensitive** escalation policy.  
- **Risk fires even though retrieval is strong.**

**Step 4 — Decision**  
- **`status = escalated`** — because **policy overrides good retrieval**.

**Step 5 — Response**  
- User sees **escalation template**, not an invented admin override.

**Why this is correct behavior**

> **Strong documentation match does not grant permission to bypass org-admin boundaries.** Your system separates **“can we find text?”** from **“should we act?”** — that is **production-grade** thinking.

---

## 5. ⚖️ Design Decisions (INTERVIEW GOLD)

| Decision | What you chose | Why | Alternatives | Why rejected |
|----------|----------------|-----|--------------|--------------|
| **BM25 vs dense embeddings** | BM25 (lexical) | Fast, tiny deps (stdlib), interpretable token overlap, works well on FAQ-like text | Sentence-BERT, OpenAI embeddings | Heavier, often needs GPU/tuning; adds non-determinism if API-backed; harder to audit |
| **Deterministic vs LLM** | Rules + BM25 only | Reproducible runs; auditable; no API keys at runtime | GPT-4 triage | Cost, latency, drift, harder to prove “no hallucination” |
| **Rule-based risk vs ML classifier** | Explicit phrases + regex boundaries | Explainable (“we escalated because phrase X”); no training data dependency | Toxicity classifier, fine-tuned BERT | Needs labeled data; opaque errors; domain shift |
| **Escalation-first** | If unsure or sensitive → escalate | Prevents wrong authoritative answers on billing/legal/account topics | Always answer from top doc | Unsafe; wrong confidence |
| **Overlap filter before BM25** | Require minimum query–doc token overlap | Reduces junk high-idf matches | No filter | More noisy top hits |
| **Company-aware retrieval** | Filter chunks by inferred company | Keeps Visa vs HackerRank vs Claude separation | Single global index only | Cross-domain false positives |
| **YAML / front matter parsing** | Line-based closing `---` | Fixes **`---` inside URLs** breaking naive splits | Naive `split("---")` | Corrupted chunks / leaked YAML into answers |
| **Snippet extraction** | Sentences overlapping query tokens | Keeps answer tied to retrieved evidence | Summarize with LLM | Introduces hallucination risk |

**Soundbite for interviews:**

> **“We optimized for auditability and safety: same ticket → same decision, and every escalation has a named reason.”**

---

## 6. 🚫 Failure Cases & Limitations

| Failure mode | What happens | Example |
|----------------|--------------|---------|
| **Vague ticket** | Low BM25 + low overlap → **escalate** or weak reply | “It’s not working” with no product name |
| **Novel phrasing** | BM25 is lexical — synonyms may miss the best article | “employment verification” vs “background check” wording |
| **Multilingual** | Rules tuned on phrases; retrieval depends on token overlap | French ticket: partial phrase lists + encoding quirks |
| **Right doc, wrong chunk** | Chunk boundaries may drop the best sentence | Mitigated by overlapping chunks |
| **Legitimate question flagged risky** | Conservative rules → **over-escalation** | Rare support FAQs touching “dispute” |
| **Benign `invalid`** | Safe refusal vs escalate depends on policy | Destructive prompts should still escalate via risk |

### How you’d improve (honest interview answer)

1. **Query expansion** with a **small synonym map** per domain (still deterministic).  
2. **Lightweight reranker** (cross-encoder) **offline** — still no API at runtime if bundled.  
3. **Per-locale** risk phrases.  
4. **Calibration table** linking BM25 scores to precision on a **labeled validation slice**.  
5. **Human-in-the-loop** logging to refine rules after seeing false positives.

---

## 7. 🛡️ Safety & Escalation Philosophy

**Why escalation matters**

- Support isn’t trivia — wrong answers about **money, access, or legal** erode trust and create liability.

**How you detect risk**

- Curated **phrase lists** for sensitive intents + **word-boundary** matching for short tokens.  
- Separate **outage / platform-wide** wording from normal bugs.

**Why avoid hallucination**

- User-facing text is either **snippet-derived** or **fixed templates** — no free-form synthesis.

**When escalation beats answering**

- Account-specific actions, manipulation of scores, merchant disputes, identity theft, vulnerability disclosure, probing for internal logic — **always human**.

**One memorable line:**

> **“We’d rather miss an auto-reply than ship a confident wrong policy.”** *(Tune this tone to your interviewer.)*

---

## 8. 📊 Evaluation Strategy

| Dimension | What you did |
|-----------|----------------|
| **Schema** | Validate column names, allowed enums, row count = input count |
| **Grounding** | Responses trace to **top chunk** sentences; no LLM paraphrase |
| **Balance** | Check **not all escalated** / **not all replied** when dataset is mixed |
| **Logs** | Each ticket: **status**, **score**, **risk**, **doc id** |
| **Regression** | Same corpus + CSV → **bit-identical** output (determinism) |

**Honesty note for judges:** Automated checks cannot prove semantic correctness — you relied on **design** (snippets + escalate) plus **spot checks**.

---

## 9. 🎯 Interview Questions & Perfect Answers

Each item: **Question → Simple answer → Advanced answer → Remember**.

### Q1 — Explain your system in 30 seconds.

- **Simple:** We search local help articles with BM25, check safety rules, then either quote relevant sentences or escalate.  
- **Advanced:** Lexical index over chunked Markdown with YAML-aware parsing; company filtering; BM25 with overlap gating; declarative risk layer; deterministic decision DAG; snippet-only generation.  
- **Remember:** **Lookup + policy**, not generation.

### Q2 — Why no LLM?

- **Simple:** Predictability, traceability, and no dependency on external APIs or keys.  
- **Advanced:** LLMs optimize fluency, not faithfulness; evaluation wants **grounded** behavior; stdlib-only deployment is simpler to reproduce.  
- **Remember:** **Faithfulness > fluency** for this challenge.

### Q3 — Why BM25 instead of embeddings?

- **Simple:** BM25 is classic search — fast, explainable, works on FAQs.  
- **Advanced:** Sparse retrieval aligns with keyword-heavy support text; avoids embedding model versioning; ties naturally to **token overlap** filters.  
- **Remember:** **Interpretability + zero extra models.**

### Q4 — What if BM25 returns the wrong document?

- **Simple:** We escalate when scores are weak or rules say so.  
- **Advanced:** Overlap filter + score threshold + risk overrides; logging `doc_id` for debugging.  
- **Remember:** **Thresholds encode humility.**

### Q5 — How do you handle multilingual tickets?

- **Simple:** Partially — phrase lists help; retrieval still token-based.  
- **Advanced:** Word-boundary fix for `fraud` vs `fraude`; expand locale-specific phrases over time.  
- **Remember:** **Honest limitation + roadmap.**

### Q6 — Isn’t rule-based risk brittle?

- **Simple:** Yes, but it’s **transparent** and easy to fix.  
- **Advanced:** Trade opacity of ML for maintainability; rules map to compliance requirements; iterate from logs.  
- **Remember:** **Explicit > opaque.**

### Q7 — What’s your escalation philosophy?

- **Simple:** When in doubt, human.  
- **Advanced:** Risk triggers for account/legal/security; weak retrieval; ambiguous org context — structured reasons in CSV/log.  
- **Remember:** **Conservative is a feature.**

### Q8 — How do you prevent hallucination?

- **Simple:** We only stitch sentences from the retrieved chunk.  
- **Advanced:** No free-form NLG; escalation template for everything else; HTML unescape only.  
- **Remember:** **Evidence-bound generation.**

### Q9 — Tradeoffs of your approach?

- **Simple:** Safer and deterministic, but less “creative” with synonyms.  
- **Advanced:** Precision/recall tradeoff; over-escalation cost vs wrong-answer cost; maintenance of phrase lists.  
- **Remember:** **You chose operational risk profile.**

### Q10 — How would you scale to 10M chunks?

- **Simple:** Better indexing and hardware — maybe Elasticsearch.  
- **Advanced:** Inverted index sharding; caching BM25 statistics; prefilter by company/topic; batch offline rebuilds.  
- **Remember:** **Engineering scaling, not model scaling.**

### Q11 — How would you add an LLM later?

- **Simple:** Only where it can cite sources — like reranking or quoting.  
- **Advanced:** LLM-as-reranker with **forced span selection** from top-k chunks; refuse if no span exceeds confidence; same escalate paths.  
- **Remember:** **Constrained LLM**, not open chat.

### Q12 — What was hardest bug you solved?

- **Simple:** Front matter parsing broke when URLs had `---` inside them.  
- **Advanced:** Naive `split("---")` cut YAML at wrong delimiter; switched to **line-based** closing delimiter; restored clean chunks.  
- **Remember:** **Real-world Markdown is messy.**

### Q13 — How did you validate outputs?

- **Simple:** Row counts, allowed values, spot-read logs.  
- **Advanced:** Schema validation script; manual edge-case tickets (fraud, outage, None company); determinism check.  
- **Remember:** **Mix of automated + spot.**

### Q14 — What metrics would you track in production?

- **Simple:** Escalation rate, time-to-resolve for escalated, thumbs-down on replies.  
- **Advanced:** Precision@k on retrieval labels; confusion matrix on reply vs escalate; rule-level false positive rates.  
- **Remember:** **Outcome + intermediate retrieval quality.**

### Q15 — How does company inference fail?

- **Simple:** Short vague tickets might not mention a product.  
- **Advanced:** Fallback `None` + weaker retrieval triggers escalation rule — avoids wrong-domain answers.  
- **Remember:** **Ambiguity → conservative path.**

### Q16 — Why snippet concatenation instead of summarization?

- **Simple:** Summaries invent wording; snippets don’t.  
- **Advanced:** Minimizes hallucination surface; sentences remain tied to source chunk.  
- **Remember:** **Faithfulness constraint.**

### Q17 — What about PII in logs?

- **Simple:** Logs should avoid storing full credit cards; truncate previews.  
- **Advanced:** Redaction pipeline for production; configurable log level.  
- **Remember:** **Show security awareness.**

### Q18 — Could an attacker jailbreak your agent?

- **Simple:** Risk phrases catch many probes; otherwise escalate.  
- **Advanced:** No LLM means fewer prompt-injection surfaces; policy layer still critical.  
- **Remember:** **Defense in depth.**

### Q19 — How did you use AI coding tools?

- **Simple:** AI helped structure modules and draft boilerplate — I reviewed every rule and threshold.  
- **Advanced:** Treat AI as **accelerator**, not authority; validated determinism and grounding manually; took ownership of safety policy.  
- **Remember:** **Human accountability narrative.**

### Q20 — What would you redo with more time?

- **Simple:** Better evaluation dataset + synonym expansion.  
- **Advanced:** Calibrated scores per domain; gold labels for retrieval; A/B on thresholds; richer multilingual rules.  
- **Remember:** **Iteration mindset.**

---

## 10. 🧠 My Unique Edge

Use **3–5** of these (adapt wording):

1. **“We separated retrieval confidence from policy permission — good BM25 scores don’t override safety.”**  
2. **“The system is reproducible: same ticket, same output — that’s rare once LLMs enter the loop.”**  
3. **“Every escalation has an explicit reason string — built for audits, not demos.”**  
4. **“We fixed real Markdown edge cases (YAML delimiters inside URLs) — production corpus hygiene matters.”**  
5. **“I optimized for ‘boring reliability’: interpretable rules beat flashy generations in regulated support contexts.”**

---

## 11. 🗣️ Speaking Practice Section

### 60-second pitch

> “I built a deterministic support triage pipeline over a local Markdown corpus. Tickets are searched with BM25 over chunked articles with company-aware filtering. A separate rule layer catches sensitive or adversarial requests — things like account access, disputes, or outages — and escalates those regardless of retrieval strength. When it’s safe and well-supported, we answer by extracting sentences directly from the top-matching chunk so we stay grounded. Everything writes to CSV with audit-friendly justifications and logs with scores and document IDs. No LLM at runtime — reproducibility and traceability were the goals.”

### 2-minute explanation

Expand the 60-second version with:

- **Parsing:** YAML breadcrumbs → rich `product_area`; careful front matter handling.  
- **Retrieval:** BM25 + overlap threshold + top-k + company filter.  
- **Decision DAG:** risk → empty hits → score threshold → invalid handling.  
- **Outputs:** schema + logging for debugging.

### 5-minute deep dive

Add:

- Walk through **one escalation** and **one reply** example.  
- Discuss **tradeoffs** (BM25 vs embeddings; rules vs ML).  
- Close with **limitations** + **next iteration** (synonym map, reranker, calibration).

---

## 12. 🚀 Final Tips Before Interview

| Do | Don’t |
|----|--------|
| Open your repo and **trace one ticket** in code as rehearsal | Claim “zero hallucination” without nuance |
| Prepare **two failure stories** with fixes | Trash-talk LLMs — show **when** you’d add them |
| State **assumptions** explicitly | Overpromise multilingual accuracy |
| Tie answers to **business risk** (trust, compliance) | Memorize jargon without meaning |

**If you don’t know:**  
> “We didn’t optimize that path — here’s how I’d measure it and what data I’d need.” **That answer scores higher than bluffing.**

---

## Appendix — How this was built (coding journey)

<details>
<summary><strong>For “walk me through how you built it”</strong></summary>

1. **Define success:** Grounded answers, explicit escalation, deterministic CSV, no runtime network.  
2. **Corpus first:** Recursive `.md` load; discover YAML breaking cases (`---` in URLs).  
3. **Chunk + index:** BM25 statistics over chunks; stable sort for ties.  
4. **Retrieval API:** `search(query, company)` with overlap filter.  
5. **Taxonomy:** Request types + risk lists (iterate using sample tickets).  
6. **Decision layer:** Encode “risk beats retrieval.”  
7. **Generation:** Sentence overlap scoring; strip boilerplate where possible.  
8. **Validation:** Row counts + enums + spot logs.  
9. **Packaging:** Zip `code/` only; document run command in README.

</details>

---

**You’ve got this.** Your story is **coherent**, **defensible**, and **honest** — that reads as senior signal in interviews.

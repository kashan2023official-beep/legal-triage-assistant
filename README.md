# ⚖️ Legal Triage Assistant — Complete Project Documentation

> A production-style RAG pipeline for first-pass legal triage of U.S. civil matters. Classifies user intent, extracts structured intake facts, retrieves relevant Q&A guidance and state statutes via hybrid dense + sparse search, and produces a strictly-grounded response with clarification questions.

**Status:** Working prototype · **Language:** Python 3.12 · **Env:** Kaggle (CPU embeddings + 4-bit LLM on GPU) · **License:** MIT

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Repository Layout](#4-repository-layout)
5. [Data Stores](#5-data-stores)
6. [Models](#6-models)
7. [Pipeline Stages (Deep Dive)](#7-pipeline-stages-deep-dive)
   - 7.1 [Intent Classification](#71-intent-classification)
   - 7.2 [Query Translation / Enrichment](#72-query-translation--enrichment)
   - 7.3 [Intake State Extraction](#73-intake-state-extraction)
   - 7.4 [Hybrid Retrieval](#74-hybrid-retrieval)
   - 7.5 [Grounded Response Generation](#75-grounded-response-generation)
8. [Normalization Utilities](#8-normalization-utilities)
9. [Prompt Engineering Details](#9-prompt-engineering-details)
10. [Retrieval Scoring Internals](#10-retrieval-scoring-internals)
11. [Safety & Guardrails](#11-safety--guardrails)
12. [Configuration Reference](#12-configuration-reference)
13. [Installation & Setup](#13-installation--setup)
14. [Running the Assistant](#14-running-the-assistant)
15. [Walkthrough Example](#15-walkthrough-example)
16. [Design Decisions & Rationale](#16-design-decisions--rationale)
17. [Failure Modes & Known Limitations](#17-failure-modes--known-limitations)
18. [Extending the System](#18-extending-the-system)
19. [Performance Notes](#19-performance-notes)
20. [Ethics, Legal & Disclaimers](#20-ethics-legal--disclaimers)
21. [Appendix: Function Index](#21-appendix-function-index)

---

## 1. Executive Summary

The Legal Triage Assistant is a conversational AI that performs **structured intake and triage** for civil legal matters in the United States. It is **not** a lawyer, and it does not give legal advice — instead it:

- Listens to a user's natural-language description of a problem.
- Classifies the message as a legal question, clarification answer, chit-chat, or off-topic request.
- Rewrites the message into a clean search query.
- Extracts structured facts (state, case type, amount, contract status, payment status).
- Retrieves relevant **practical guidance** and **state statutes**.
- Produces a grounded response with an initial assessment, actionable next steps, and clarification questions.

The system was designed to **never hallucinate a statute**. If retrieval fails to surface a genuinely relevant statute, the LLM is instructed to say so explicitly.

---

## 2. Problem Statement

People facing civil legal problems often don't know:

- Whether their situation is even a legal issue.
- Which jurisdiction's law applies.
- What statutes or remedies exist.
- What to do next.

Existing tools either (a) give generic advice, (b) hallucinate citations, or (c) require expensive legal consultation. This project explores **structured retrieval-augmented triage** as a middle ground: give the user meaningful orientation without pretending to be a lawyer, and **never fabricate authority**.

---

## 3. High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         USER MESSAGE                          │
└────────────────────────────┬──────────────────────────────────┘
                             │
                ┌────────────▼─────────────┐
                │  1. Intent Classifier    │
                │  (LLM → 4-way label)     │
                └────────────┬─────────────┘
                             │
      ┌──────────────────────┼───────────────────────┐
      │                      │                       │
  off_topic /          new_legal_question     clarification_answer
  chit_chat                 │                       │
      │                     ▼                       ▼
      │         ┌────────────────────┐   ┌─────────────────────┐
      │         │ 2a. Translate Query│   │ 2b. Enrich Query    │
      │         │ (standalone rewrite)│  │ (merge prev + new)  │
      │         └─────────┬──────────┘   └──────────┬──────────┘
      │                   └────────────┬────────────┘
      │                                │
      │                   ┌────────────▼────────────┐
      │                   │ 3. Intake State Extractor│
      │                   │ (delta-only JSON)        │
      │                   │ + regex fallbacks        │
      │                   └────────────┬────────────┘
      │                                │
      │                   ┌────────────▼────────────┐
      │                   │ 4. Hybrid Retrieval      │
      │                   │  ├─ QA FAISS (bge-small) │
      │                   │  └─ Statutes:            │
      │                   │     dense + BM25 → RRF   │
      │                   │     → soft-score filter  │
      │                   └────────────┬────────────┘
      │                                │
      │                   ┌────────────▼────────────┐
      │                   │ 5. Grounded Generation   │
      │                   │ (strict-prompt LLM)      │
      │                   └────────────┬────────────┘
      ▼                                ▼
  Canned reply            Assessment + Actions + Clarifications
```

---

## 4. Repository Layout

```
legal-triage-assistant/
├── README.md                       # Overview and quick start
├── LICENSE                         # MIT
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes data stores + weights
├── main.py                         # Entry point
│
├── src/
│   ├── __init__.py                 # Public API re-exports
│   ├── config.py                   # Paths, model IDs, tunables
│   └── legal_triage.py             # Full pipeline (~700 LOC)
│
├── notebooks/
│   └── legal_triage.ipynb          # Notebook runner
│
├── docs/
│   └── ARCHITECTURE.md             # Deep-dive design notes
│
└── data/
    ├── .gitkeep                    # Placeholder
    ├── qa_faiss_store/             # (gitignored) Q&A index + docs
    └── statutes_faiss_store/       # (gitignored) per-state indices
```

---

## 5. Data Stores

Two prebuilt FAISS stores power retrieval. Both are **mounted read-only** from Kaggle Datasets at runtime and are excluded from git.

### 5.1 Q&A FAISS Store

| File | Contents |
|---|---|
| `qa_index.faiss` | FAISS index, **3,742 vectors**, dim = 384 |
| `qa_documents.pkl` | Python list of dicts; each has `metadata.legal_guidance` and optionally `metadata.state` |

**Purpose:** generic practical guidance ("what people usually do in this situation"). **Never** treated as a citation source.

### 5.2 Per-State Statutes FAISS Store

| File | Contents |
|---|---|
| `manifest.json` | List of jurisdiction codes (e.g. `["ak","al",...,"federal",...]`) |
| `state_<code>_index.faiss` | Dense index for that state's statutes |
| `state_<code>_docs.pkl` | Doc metadata: `citation`, `section_title`, `full_text`, `state` |
| `state_<code>_bm25.pkl` | Pickled BM25 object with tokenized corpus |

**Jurisdictions loaded:** 51 including all 50 states, `federal`, `dc`, `pr`.

**Corpus sizes (approximate):**

| State | Vectors | State | Vectors |
|---|---|---|---|
| CA | 161,525 | TX | 119,720 |
| MS | 156,455 | IN | 75,626 |
| IL | 72,432 | NJ | 55,985 |
| WA | 51,104 | NV | 47,899 |
| federal | 46,532 | LA | 43,223 |
| AL | 41,895 | MI | 40,519 |
| NY | 40,140 | SD | 39,570 |
| MD | 38,182 | … | … |

**Load-time guardrails:**

- Every FAISS index is reconstructed and **L2-normalized in memory** and rebuilt as `IndexFlatIP`. This guarantees `IndexFlatIP` score == cosine similarity, regardless of how the store was originally built. The original index is discarded.
- `assert index.ntotal == len(documents)` for both Q&A and each state — corruption fails fast.

---

## 6. Models

| Model | Role | Device | Size | Notes |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | Text embeddings | CPU | ~133 MB | Uses BGE's query prefix `"Represent this sentence for searching relevant passages: "` |
| `NousResearch/Hermes-3-Llama-3.1-8B` | Generation | GPU | ~5 GB (4-bit NF4) | Loaded with `bitsandbytes` 4-bit NF4, double quant, `float16` compute |

**Total VRAM footprint:** ~6 GB. Fits comfortably on a T4 (16 GB) or P100.

---

## 7. Pipeline Stages (Deep Dive)

### 7.1 Intent Classification

**Function:** `classify_input_intent(user_input, last_assistant_message, last_query)`

**Labels:**

| Label | Meaning |
|---|---|
| `new_legal_question` | User asks a new legal question or describes a new legal situation |
| `clarification_answer` | User is answering a clarification question OR adding a factual detail about the current case |
| `off_topic` | Request unrelated to legal triage (essays, coding, recipes, trivia) |
| `chit_chat` | Greetings, thanks, small talk, filler |

**Method:** LLM with strict JSON-only prompt. The model sees the last assistant message, the last substantive query, and the new user input.

**Fallback (if JSON parse fails):** Heuristic — if the message starts with any of `what/how/why/when/where/who/can i/should i/is it/do i/does`, classify as `new_legal_question`; otherwise `clarification_answer`.

**Downstream effect:**

- `off_topic` / `chit_chat` → canned reply, loop continues.
- `new_legal_question` → fresh query translation.
- `clarification_answer` → query enrichment against prior query.

---

### 7.2 Query Translation / Enrichment

**Function:** `translate_query(raw_user_input, last_assistant_message, previous_query=None)`

**Purpose:** Rewrite the raw message into a **standalone** legal search query that captures the user's intent without relying on context.

**Rules enforced by the prompt:**

- If a `Previous Legal Query` is provided and the user is *adding a fact* (not asking a new question), produce a **single merged query**.
- If the user is answering a clarification question, use the previous assistant message only to resolve references like "yes", "no", "that one".
- **Do not invent** a jurisdiction, statute number, or party name the user did not mention.
- **Do not copy** action items, procedural suggestions, or statute names out of the previous assistant message.

**Example:**
```
Previous Legal Query: "Is there a legal basis for someone suing me over a breach of an oral contract?"
Raw User Input:       "I am in arizona"
→ "Is there a legal basis for someone suing me over a breach of an oral contract in Arizona?"
```

---

### 7.3 Intake State Extraction

**Function:** `update_intake_state(user_query, current_state, state_indices, last_assistant_message)`

**Schema (`_ALLOWED_KEYS`):**

| Field | Type | Extraction Rule |
|---|---|---|
| `case_type` | str | Short classification (e.g. `"contract_dispute"`). **Only field inferred from topic.** |
| `state` | 2-letter code / `federal` | Must be **explicitly named**. Never inferred from a city, court, area code, or time zone. |
| `disputed_amount` | float (USD) | Numeric value only. `5000`, not `"5000 dollars"`. |
| `written_contract_exists` | bool | Explicit only. |
| `payment_status` | enum | `unpaid` / `partially_paid` / `paid` / `disputed`. |

**Critical design: delta-only extraction.**

The LLM is told:

> NEVER echo back a value from a previous turn. Output a DELTA, not a full state.
> If the message contains no new extractable facts, output exactly: `{}`

This prevents the model from drifting into stale state, hallucinating values, or overwriting facts the user already provided.

**Validation pipeline:**

1. Strip markdown fences from the LLM output.
2. `json.loads()`.
3. `_validate_delta()` — whitelist keys, normalize each value:
   - `case_type` → lowercase, stripped
   - `state` → through `normalize_state()`
   - `disputed_amount` → through `parse_amount()`
   - `written_contract_exists` → bool coercion
   - `payment_status` → enum check
4. Merge into `current_state` — **only valid keys are merged**; nothing is dropped from existing state.

**Regex fallbacks (`_apply_fallbacks`):**

| Fallback | Regex | Purpose |
|---|---|---|
| `fallback_extract_state` | `_STATE_REGEX` | Catches full state names ("texas", "state of arizona") that the LLM may have missed |
| `fallback_extract_amount` | `_AMOUNT_REGEX` | Catches `$5,000`, `5000 dollars`, `1.5m`, `bucks` |

Fallbacks only fire when the LLM did not populate the field. This is a **deterministic safety net**.

---

### 7.4 Hybrid Retrieval

Two retrievals happen per turn: Q&A retrieval (dense only) and statute retrieval (dense + BM25 with RRF fusion).

#### 7.4.1 Q&A Retrieval

```
search_string = f"Area of law: {case_type}. Situation: {clean_query}"
                │
                ▼
     bge-small-en-v1.5 (with query prefix)
                │
                ▼
     FAISS IndexFlatIP → top-(top_k_qa * 6)
                │
                ▼
     state filter: keep doc if doc_state == state_code or doc has no state
                │
                ▼
     top-2 Q&A docs → "Practical Guidance: ..."
```

If the user's state is unknown, docs with a specific state tag are dropped.

#### 7.4.2 Statute Retrieval

**Function:** `_retrieve_statutes(state_code, clean_query, case_type, top_k=3, pool=40)`

**Step 1 — Domain template expansion:**
```python
_CASE_QUERY_TEMPLATES = {
    "contract_dispute": "breach of contract, suit on account, open account, recovery of unpaid debts, damages, and attorney's fees",
    "landlord_tenant":  "landlord tenant law, eviction, lease disputes, and recovery of rent",
    "employment":       "employment law, unpaid wages, wrongful termination, wage claims",
    "personal_injury":  "personal injury, negligence, tort damages",
    "family":           "family law, divorce, child custody, child support, spousal support",
    "consumer":         "consumer protection, deceptive trade practices, warranty claims",
    "insurance":        "insurance claims, coverage disputes, insurer obligations",
}
```

The final statute query:
```
"{STATE} statutes on {domain_phrase}. {clean_query}"
```

**Step 2 — Dense search:**
- Embed with BGE prefix
- `IndexFlatIP` search → top-`pool` (40) by cosine

**Step 3 — BM25 search:**
- Tokenize query
- Filter out tokens not in the BM25 corpus vocabulary (prevents OOV noise)
- `bm25.get_scores(tokens)` → `np.argsort` → top-`pool` (40)

**Step 4 — RRF fusion:**

```python
def _rrf_fuse(vec_rank, bm25_rank, k=60):
    scores = {}
    for rank, idx in enumerate(vec_rank):
        scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_rank):
        scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=-score)
```

Reciprocal Rank Fusion is **scale-free** — it doesn't matter that cosine ∈ [0,1] and BM25 scores are unbounded. Only ranks matter.

**Step 5 — Filtering:**

| Filter | Threshold | Rationale |
|---|---|---|
| BM25-only candidates | must be in BM25 top-5 | Reject sparse outliers |
| Dense candidates | cosine ≥ 0.45 | Reject weak semantic matches |
| Soft score | ≥ -1.0 | Allow mild anti-signals |

**Step 6 — Soft re-ranking:**

```python
score = 0
for signal in _CASE_SIGNALS[case_type]:
    if signal in header: score += 2.0
    elif signal in body: score += 0.5
for anti in _CASE_ANTI_SIGNALS[case_type]:
    if anti in header: score -= 3.0
    elif anti in body: score -= 0.3
for anti in _CASE_CODE_ANTI[case_type]:
    if anti in citation: score -= 5.0
```

- **Header** = `citation + citation_short + section_title`
- **Body** = `full_text`
- **Code-level anti-signals** carry the heaviest penalty (-5.0). If the citation contains e.g. `"insurance code"` for a `contract_dispute`, it's suppressed immediately.

**Step 7 — Sort and return:**
```python
scored.sort(key=lambda x: (-soft_score, -rrf_score))
return scored[:top_k]  # default top 3
```

---

### 7.5 Grounded Response Generation

**Function:** `get_legal_triage(clean_query, current_state, top_k_qa=2, top_k_statutes=3)`

**Inputs assembled into the prompt:**

| Block | Source |
|---|---|
| Citation rule | `_build_citation_rule(state_code, n_candidates, corpus_loaded)` |
| Known-facts guard | Built from `current_state` — "do NOT ask again" list |
| Known Facts JSON | `json.dumps(current_state, indent=2)` |
| Retrieved Q&A Guidance | Top-2 docs |
| Retrieved Statutes | Top-3 docs, formatted as `Citation / State / Title / Text` |

**Dynamic citation rule** — this is the anti-hallucination heart of the system. Four cases:

1. **No state** → "Ask the user which US state they are in."
2. **State known, no corpus loaded** → "Say the corpus is unavailable."
3. **State known, corpus loaded, zero candidates** → "Say no relevant statute was found in the {STATE} corpus. Do NOT ask for the state."
4. **State known, candidates exist** → "(a) cite one if it genuinely applies; (b) if none match, say so; (c) do NOT stretch."

**Output format enforced:**

```
### Initial Assessment
<1-2 short paragraphs>

### Action Steps You Can Take Now
<numbered list>

### Clarification Questions
- <question 1>
- <question 2>
OR
(none — all required facts are known)
```

**Generation settings:** `do_sample=False` (greedy), `max_new_tokens=600`, `pad_token_id=tokenizer.eos_token_id`.

---

## 8. Normalization Utilities

### 8.1 State Normalization

**Function:** `normalize_state(raw, valid_codes)`

Handles:

| Input | Output |
|---|---|
| `"texas"` | `"tx"` |
| `"TX"` | `"tx"` |
| `"State of Texas"` | `"tx"` |
| `"United States"` | `"federal"` |
| `"District of Columbia"` | `"dc"` |
| `"washington dc"` | `"dc"` |
| `"Narnia"` | `None` |

Lookup chain: exact code → full name → substring partial match.

### 8.2 Amount Parsing

**Function:** `parse_amount(raw)`

Handles:

| Input | Output |
|---|---|
| `5000` | `5000.0` |
| `"$5,000"` | `5000.0` |
| `"5000 dollars"` | `5000.0` |
| `"1.5k"` | `1500.0` |
| `"2m"` | `2000000.0` |
| `"five thousand"` | `5000.0` (via `word2number`) |
| `"-100"` | `None` (must be positive) |
| `"not a number"` | `None` |

---

## 9. Prompt Engineering Details

### Prompt philosophy

Every prompt follows the same pattern:

1. **Role declaration** — "You are a strict X."
2. **Absolute rules** — numbered, explicitly called "failures" if violated.
3. **Concrete examples** — few-shot with `User: ... Output: ...`.
4. **Output format constraints** — "Respond with ONLY a JSON object. No markdown."

### Anti-hallucination techniques used

| Technique | Where |
|---|---|
| Delta-only extraction | `update_intake_state` |
| "Never infer a state from a city" | `update_intake_state` |
| "Never echo a value from a previous turn" | `update_intake_state` |
| Dynamic citation rule | `get_legal_triage` |
| "Do NOT copy statute citations out of Q&A guidance" | `get_legal_triage` |
| "Better no citation than a wrong one" | `_build_citation_rule` |
| Known-facts guard list | `get_legal_triage` |
| "Do NOT invent jurisdiction / statute / party name" | `translate_query` |

### Safety rules in the system prompt

The final generation prompt includes:

> 9. If the user asks to conceal assets, evade taxes, destroy evidence, or commit illegal acts, refuse that request and outline lawful alternatives only.

Plus additional constraints:

- Do not assign gender unless stated by user.
- Do not suggest "on attorney letterhead" unless user said they are an attorney.
- Only mention mechanics lien if construction is involved.
- Do not reference case/docket numbers or party names unless mentioned.

---

## 10. Retrieval Scoring Internals

### 10.1 Reciprocal Rank Fusion (RRF)

```
score(doc) = Σ   1 / (k + rank_i + 1)
            i

k = 60 (default, config.RRF_K)
```

Why RRF?

- **Scale-free:** dense cosine and BM25 scores are incomparable; RRF only uses ranks.
- **Robust:** a document ranked #1 in one and #40 in the other still gets a nonzero fused score.
- **Standard:** widely used in hybrid search literature.

### 10.2 Soft Scoring Signals

For `contract_dispute`, `_CASE_SIGNALS` includes:
`breach of contract`, `suit on account`, `open account`, `sworn account`,
`contract debt`, `unpaid invoice`, `attorney's fees`, `mechanic's lien`,
`recovery of debt`, `damages`, `remedies`, `contract`, `debt`.

For `contract_dispute`, `_CASE_CODE_ANTI` includes:
`insurance code`, `transportation code`, `natural resources code`,
`local government code`, `health and safety code`, `penal code`,
`tax code`, `family code`, `government code`, `property code`, and more.

This effectively tells the retrieval system:

> For a contract dispute, a `Property Code` section that happens to contain "contract" is likely a false positive. Penalize it heavily.

### 10.3 Why Three Filters?

- **BM25-only rank ≤ 5** — BM25 alone can surface spurious matches on rare tokens. Restrict to genuinely high-ranked ones.
- **Cosine ≥ 0.45** — dense-only candidates with weak similarity are noise.
- **Soft ≥ -1.0** — allow mild anti-signals but not overwhelming ones.

---

## 11. Safety & Guardrails

| Guardrail | Implementation |
|---|---|
| No hallucinated statutes | Dynamic citation rule + candidate count signal |
| No inferred jurisdiction | Explicit rule in intake prompt |
| No inferred dollar amounts | Regex + numeric parsing on validated deltas |
| No re-asking known facts | Known-facts guard block |
| No gender assumption | Explicit prompt rule |
| No unrequested "attorney letterhead" advice | Explicit prompt rule |
| Illegal act refusal | Explicit prompt rule #9 |
| State-corpus mismatch handling | `_qa_state_ok` filter + `corpus_loaded` flag |
| Corrupt store detection | `assert ntotal == len(docs)` at load |

---

## 12. Configuration Reference

All tunables live in `src/config.py` and are env-overridable.

| Constant | Default | Purpose |
|---|---|---|
| `QA_DIR` | `./data/qa_faiss_store` | Q&A store path (`$QA_DIR`) |
| `STATUTES_DIR` | `./data/statutes_faiss_store` | Statutes store path (`$STATUTES_DIR`) |
| `EMBED_MODEL_ID` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `LLM_MODEL_ID` | `NousResearch/Hermes-3-Llama-3.1-8B` | Generator |
| `TOP_K_QA` | `2` | Q&A docs returned |
| `TOP_K_STATUTES` | `3` | Statute candidates returned |
| `RETRIEVAL_POOL` | `40` | Candidates fetched before filtering |
| `RRF_K` | `60` | RRF smoothing constant |
| `MIN_COS` | `0.45` | Minimum cosine for dense candidates |
| `MIN_SOFT_SCORE` | `-1.0` | Minimum soft score |
| `MIN_BM25_ONLY_RANK` | `5` | BM25-only candidates must be top-5 |
| `MAX_NEW_TOKENS_TRANSLATE` | `140` | Query translation budget |
| `MAX_NEW_TOKENS_INTAKE` | `140` | Intake extraction budget |
| `MAX_NEW_TOKENS_INTENT` | `60` | Intent classification budget |
| `MAX_NEW_TOKENS_RESPONSE` | `600` | Final response budget |
| `LOG_FILE` | `legal_triage.log` | Log output (`$LEGAL_TRIAGE_LOG`) |

---

## 13. Installation & Setup

### Option A — Local / Colab

```bash
git clone <your-repo-url>
cd legal-triage-assistant
pip install -r requirements.txt
```

Then either:

```bash
# Option 1: place stores under ./data/
mkdir -p data/qa_faiss_store data/statutes_faiss_store
# copy your files here ...

# Option 2: point env vars at existing paths
export QA_DIR=/path/to/qa_faiss_store
export STATUTES_DIR=/path/to/statutes_faiss_store
```

### Option B — Kaggle

1. Create a new notebook.
2. Attach the datasets:
   - `kashanali446/qa-faiss-store`
   - `kashanali446/statutes-faiss-store`
3. Enable GPU accelerator (T4 or P100).
4. Set `QA_DIR` and `STATUTES_DIR` to the mount paths.

---

## 14. Running the Assistant

### Interactive

```bash
python main.py
```

### Notebook

Open `notebooks/legal_triage.ipynb` and run.

### Programmatic

```python
from src.legal_triage import load_resources, get_legal_triage

load_resources()

state = {"case_type": "contract_dispute", "state": "az",
         "disputed_amount": 5000.0, "written_contract_exists": True,
         "payment_status": "unpaid"}

response = get_legal_triage(
    "Someone in Arizona owes me $5000 for freelance work under a written contract.",
    state,
)
print(response)
```

### Chat loop output

```
============================================================
  LEGAL ASSISTANT
Type 'quit' or 'exit' to end.
============================================================

You: Someone owes me $5000 for freelance work in Arizona.

[Search: Someone in Arizona owes me $5000 for freelance work.]
[Facts updated: {'case_type': 'contract_dispute', 'state': 'az',
                 'disputed_amount': 5000.0, 'payment_status': 'unpaid'}]

Assistant:
### Initial Assessment
...
------------------------------------------------------------
```

Debug artifacts printed inline:

- `[Search: ...]` — the translated query
- `[Facts updated: {...}]` — deltas applied this turn

---

## 15. Walkthrough Example

**Turn 1:**
```
You: can you tell me if there is a oral contract and somebody sues me over something that i promised him
```

Internally:
- Intent → `new_legal_question`
- Query → `"Is there a legal basis for someone suing me over a breach of an oral contract?"`
- Delta → `{"case_type": "contract_dispute"}`
- Retrieval: no state yet → Q&A only
- Response: no statute cited, asks for state

**Turn 2:**
```
You: I am in arizona
```

Internally:
- Intent → `clarification_answer`
- Query → `"...in Arizona?"` (merged)
- Delta → `{"state": "az"}`
- Retrieval: AZ corpus searched → candidates filtered → 0 passed
- Response: "No relevant statute was found in the AZ corpus…"

**Turn 3:**
```
You: what if there is written contract but i lost it then what happens?
```

Internally:
- Intent → `clarification_answer`
- Query → merged with "lost contract" fact
- Delta → `{"written_contract_exists": True}`
- Retrieval: AZ corpus re-searched
- Response: assessment based on guidance, clarification questions tailored

This shows **progressive intake** — each turn accumulates facts and refines the retrieval without re-asking.

---

## 16. Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **Delta-only intake** | Prevents LLM drift; makes updates auditable; guarantees existing facts are never overwritten by mistake. |
| **Regex fallbacks for state & amount** | Deterministic safety net for the two most important structured facts. |
| **RRF fusion, not score averaging** | Cosine and BM25 are on different scales; RRF is rank-based and robust. |
| **Soft scoring with anti-signals** | Pure vector search will happily retrieve an Insurance Code section for a contract dispute. Anti-signals kill these. |
| **Code-level anti-signals (-5.0)** | The strongest signal that a statute is irrelevant is the code name itself. |
| **Known-facts guard** | The LLM otherwise tends to re-ask about the state or the amount, hurting UX. |
| **Dynamic citation rule** | A single static instruction cannot handle "no state", "no corpus", "0 candidates", "candidates present". |
| **L2-normalize on load** | FAISS stores can be built incorrectly. Re-normalizing guarantees correctness at runtime. |
| **4-bit NF4 + double quant** | Fits an 8B model in ~5 GB, leaving VRAM headroom on a T4. |
| **Greedy decoding (`do_sample=False`)** | Legal triage should be deterministic and reproducible, not creative. |
| **BGE query prefix** | The `bge-small` family requires `"Represent this sentence for searching relevant passages: "` for asymmetric search. |

---

## 17. Failure Modes & Known Limitations

### Known limitations

1. **Not legal advice.** This tool provides orientation, not counsel.
2. **U.S. jurisdiction only.** The state normalization table has no non-U.S. entries.
3. **Two-turn memory.** Only `last_substantive_query` and `last_assistant_message` persist. Longer context is not tracked.
4. **Statute snapshots.** If a corpus is stale, retrievals are stale.
5. **LLM JSON fragility.** Non-JSON output silently drops non-fallback fields.
6. **Single-domain case types.** Only seven case types have tailored soft signals. Others fall back to `"legal remedies"`.
7. **No streaming.** Responses are generated fully before display.
8. **No user authentication or history.** Every run starts fresh.

### Failure modes to watch

| Symptom | Likely cause |
|---|---|
| Statute never cited | Corpus for state missing, or all candidates filtered out |
| Wrong state on record | Regex fallback matched a substring (e.g. "washington" inside a longer word) |
| Repeated clarification question | Known-facts guard not built correctly for a field |
| Empty Q&A guidance | `_qa_state_ok` filtered everything because state mismatch |
| JSON parse warnings in logs | LLM added prose around the JSON — check prompt stability |

---

## 18. Extending the System

### Add a new case type

1. Add to `_CASE_QUERY_TEMPLATES` — a keyword phrase.
2. Add to `_CASE_SIGNALS` — positive terms.
3. Add to `_CASE_ANTI_SIGNALS` — negative terms.
4. Add to `_CASE_CODE_ANTI` — code names to suppress.

### Add a new jurisdiction

Rebuild the FAISS store with a new `<code>_index.faiss` + docs + BM25 pickle, add the code to `manifest.json`, and ensure `state_indices` includes it after load.

### Add a new intake field

1. Add key to `_ALLOWED_KEYS`.
2. Handle it in `_validate_delta` with normalization.
3. Add example to the intake prompt's few-shot block.
4. Add to `known_facts_guard` in `get_legal_triage`.
5. Add to `start_legal_chat`'s initial `intake_state`.

### Swap the LLM

Change `LLM_MODEL_ID` in `config.py`. Any chat-template-compatible model (Llama, Mistral, Qwen) works — just verify the tokenizer exposes `apply_chat_template`.

### Swap the embedding model

Change `EMBED_MODEL_ID`. If you move to a model that does not require a query prefix (e.g. `e5` requires `"query: "`), update the prefix in `_retrieve_statutes` and `get_legal_triage`.

---

## 19. Performance Notes

| Operation | Cost (T4) |
|---|---|
| Model load (first call only) | ~2 min |
| Q&A embedding + search | ~50 ms |
| Statute dense + BM25 + RRF + soft rank | ~150-300 ms |
| LLM generation (intent) | ~1 s |
| LLM generation (translation) | ~2 s |
| LLM generation (intake delta) | ~2 s |
| LLM generation (final response, 600 tokens) | ~15-30 s |

**Per-turn latency (typical):** 20-40 s on a T4.

**Bottleneck:** LLM generation. If latency matters, consider:

- Smaller generator (Hermes-3-Llama-3.1-3B).
- Streamed output.
- Quantized KV cache.

---

## 20. Ethics, Legal & Disclaimers

> **This software does not constitute legal advice, does not create an attorney-client relationship, and should not be used as a substitute for consultation with a licensed attorney.**

Additional considerations:

- **Jurisdiction boundaries.** The system only models U.S. state and federal law from a fixed snapshot.
- **Bias.** Retrieval + LLM behavior reflects training data and corpus composition.
- **Privacy.** The system logs queries to `legal_triage.log`. Disable or redact in production.
- **Accountability.** Never present outputs as authoritative. Users should always confirm with a licensed attorney.

---

## 21. Appendix: Function Index

| Function | File | Purpose |
|---|---|---|
| `load_resources` | `legal_triage.py` | Load FAISS stores + models into globals |
| `start_legal_chat` | `legal_triage.py` | Interactive chat loop |
| `get_legal_triage` | `legal_triage.py` | Retrieve + generate grounded response |
| `classify_input_intent` | `legal_triage.py` | 4-way intent label |
| `translate_query` | `legal_triage.py` | Standalone query rewrite |
| `update_intake_state` | `legal_triage.py` | Delta-only fact extraction |
| `normalize_state` | `legal_triage.py` | Name → 2-letter code |
| `parse_amount` | `legal_triage.py` | String → positive float |
| `_validate_delta` | `legal_triage.py` | Whitelist + normalize delta |
| `_apply_fallbacks` | `legal_triage.py` | Regex state + amount fallback |
| `fallback_extract_state` | `legal_triage.py` | State regex |
| `fallback_extract_amount` | `legal_triage.py` | Amount regex |
| `_retrieve_statutes` | `legal_triage.py` | Hybrid statute search |
| `_rrf_fuse` | `legal_triage.py` | Reciprocal Rank Fusion |
| `_soft_score` | `legal_triage.py` | Case-type scoring |
| `_build_citation_rule` | `legal_triage.py` | Dynamic prompt fragment |
| `_normalize_index_in_memory` | `legal_triage.py` | FAISS re-normalization |
| `_generate` | `legal_triage.py` | LLM call wrapper |
| `_off_topic_response` | `legal_triage.py` | Canned off-topic/chit-chat reply |

---

## Footer

**Version:** 1.0 · **Last updated:** 2026 · **Maintainer:** `(https://github.com/kashan2023official-beep)` · **License:** MIT

> *The law is complex. This tool is a starting point, not an endpoint. Always consult a licensed attorney before acting on anything you read here.*

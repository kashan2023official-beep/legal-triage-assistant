# Architecture — Legal Triage Assistant

> Deep-dive design document for the retrieval-augmented legal triage pipeline. Focuses on the *why* behind the system's structure, not the setup steps (see `README.md` for setup).

---

## 1. High-Level Architecture

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

## 2. Pipeline Stages

### 2.1 Intent Classification

**Module:** `classify_input_intent`

**Goal:** Route the message to the right handler before incurring expensive retrieval + generation.

| Label | Handler |
|---|---|
| `new_legal_question` | Fresh query translation → full pipeline |
| `clarification_answer` | Query enrichment against previous query → full pipeline |
| `off_topic` | Canned "I can only help with legal questions" reply |
| `chit_chat` | Canned "Tell me about a legal situation" reply |

**Implementation:** LLM call with strict JSON-only prompt. If JSON parse fails, a heuristic fallback looks for interrogative stems (`what`, `how`, `why`, `when`, `where`, `who`, `can i`, `should i`, `is it`, `do i`, `does`).

**Why:** Short-circuiting off-topic and chit-chat saves ~30 seconds of retrieval + generation per turn.

---

### 2.2 Query Translation / Enrichment

**Module:** `translate_query`

**Goal:** Convert the raw user message into a standalone legal search query that captures the full intent.

**Rules enforced:**

1. If a `Previous Legal Query` is provided and the user is adding a fact, produce a single merged query.
2. If the user is answering a clarification question, use the prior assistant message only to resolve references like "yes", "no", "that one".
3. Never invent a jurisdiction, statute number, or party name.
4. Never copy action items or statute names out of the previous assistant message.

**Why merge?** Retrieval works best on a complete query. Without enrichment, a turn like *"I am in arizona"* would retrieve on just "arizona" instead of "breach of oral contract in arizona".

---

### 2.3 Intake State Extraction

**Module:** `update_intake_state`, `_validate_delta`, `_apply_fallbacks`

**Goal:** Maintain a running structured state across turns without ever overwriting facts or hallucinating new ones.

**Schema:**

| Field | Type | Extraction rule |
|---|---|---|
| `case_type` | str | Inferred from topic (only inferable field) |
| `state` | code / `federal` | Must be explicitly named |
| `disputed_amount` | float (USD) | Numeric value only |
| `written_contract_exists` | bool | Explicit only |
| `payment_status` | enum | `unpaid` / `partially_paid` / `paid` / `disputed` |

**Delta-only design:**

The LLM outputs a JSON object containing **only the fields the user explicitly stated this turn**. It is forbidden from echoing back previous state.

**Validation pipeline:**

```
LLM output
    │
    ▼
Strip markdown fences
    │
    ▼
json.loads()
    │
    ▼
_validate_delta()  ← whitelist + normalize
    │
    ▼
Merge into current_state (no overwrites with None)
```

**Regex fallbacks:** If the LLM missed `state` or `disputed_amount`, deterministic regex extractors catch them. This handles edge cases the LLM occasionally drops (e.g. `$5,000`, `1.5m`, `state of texas`).

**Why delta-only?** Prevents LLM drift. If the model were allowed to echo the full state, it would gradually add fictional facts, forget what the user said, or corrupt valid entries. Delta-only forces the model to be additive.

---

### 2.4 Hybrid Retrieval

**Modules:** `_retrieve_statutes`, `_rrf_fuse`, `_soft_score`, `_bm25_query_tokens`

Two separate retrievals happen per turn.

#### Q&A Retrieval (Dense-only)

```
"Area of law: {case_type}. Situation: {clean_query}"
                │
                ▼
      bge-small-en-v1.5 (query prefix)
                │
                ▼
      FAISS IndexFlatIP → top-12
                │
                ▼
      state filter: keep if doc_state == state_code or no tag
                │
                ▼
      top-2 → "Practical Guidance"
```

#### Statute Retrieval (Hybrid)

```
clean_query
    │
    ├─► Domain template expansion
    │      "AZ statutes on breach of contract, suit on account, ..."
    │
    ├─────────────────────────┬──────────────────────────┐
    ▼                         ▼                          
 Dense search              BM25 search                  
 (bge-small + FAISS)       (filtered tokens)            
 top-40                    top-40                       
    │                         │                          
    └──────────┬──────────────┘                          
               ▼                                         
       RRF fusion (k=60)                                 
               │                                         
               ▼                                         
    ┌──────────┴──────────┐                              
    │  Filtering          │                              
    │  - cos ≥ 0.45       │                              
    │  - BM25-only ≤ 5    │                              
    │  - soft ≥ -1.0      │                              
    └──────────┬──────────┘                              
               ▼                                         
    Soft re-rank by (-soft, -rrf)                        
               │                                         
               ▼                                         
           top-3 statutes                                
```

**Reciprocal Rank Fusion (RRF):**

```python
score(doc) = Σ 1 / (k + rank_i + 1)   over all ranked lists
```

**Why RRF and not score averaging?**

- Cosine scores lie in [0, 1]. BM25 scores are unbounded.
- Averaging would let BM25 dominate arbitrarily.
- RRF only uses ranks, so it's scale-free and robust.

**Soft scoring:**

The soft score combines:

- **Positive signals** (`_CASE_SIGNALS`) — e.g. "breach of contract", "suit on account" for contract disputes.
- **Anti-signals** (`_CASE_ANTI_SIGNALS`) — e.g. "insurance", "landlord", "criminal".
- **Code-level anti-signals** (`_CASE_CODE_ANTI`) — e.g. "insurance code", "penal code".

Weights:

| Location | Signal | Anti-signal |
|---|---|---|
| Header (citation + title) | +2.0 | -3.0 |
| Body (full text) | +0.5 | -0.3 |
| Citation code name | — | **-5.0** |

**Why code-level anti-signals are strongest:** The single most reliable signal that a statute is irrelevant is that its **code name** doesn't belong to the case type. A `Property Code` section that happens to contain the word "contract" is almost certainly a false positive for a contract dispute. The -5.0 penalty ensures it never surfaces.

---

### 2.5 Grounded Response Generation

**Module:** `get_legal_triage`, `_build_citation_rule`

**Goal:** Produce a formatted, grounded response that never hallucinates a statute.

**Inputs assembled:**

1. **Dynamic citation rule** — the core anti-hallucination mechanism.
2. **Known-facts guard** — a "do not ask again" list built from `current_state`.
3. **Known Facts JSON** — full state with `null` for unknowns.
4. **Retrieved Q&A Guidance** — top-2 docs.
5. **Retrieved Statutes** — top-3 docs.

**Dynamic citation rule — 4 cases:**

| Condition | Behavior |
|---|---|
| No state provided | Must not cite; must ask for state |
| State known, no corpus | Must not cite; must say corpus is unavailable |
| State known, 0 candidates | Must not cite; must say no relevant statute was found |
| State known, N candidates | Must cite one if genuinely relevant; must NOT stretch |

**Known-facts guard example:**

```
- State is already known: AZ — do NOT ask which state.
- Case type is already known: contract_dispute — do NOT ask what kind of case.
- Disputed amount is already known: $5000.00 — do NOT ask for the amount.
```

**Output format enforced:**

```
### Initial Assessment
<1-2 short paragraphs>

### Action Steps You Can Take Now
<numbered list>

### Clarification Questions
- <question>
OR
(none — all required facts are known)
```

**Generation settings:** Greedy (`do_sample=False`), `max_new_tokens=600`, deterministic `pad_token_id`.

**Why greedy?** Legal triage should be reproducible. Sampling introduces variance that's undesirable when verifying guardrails.

---

## 3. Data Stores

### 3.1 Q&A FAISS Store

| File | Contents |
|---|---|
| `qa_index.faiss` | 3,742 vectors, dim 384 |
| `qa_documents.pkl` | Dicts with `metadata.legal_guidance` and `metadata.state` |

**Purpose:** Generic practical guidance. **Never** a citation source.

### 3.2 Per-State Statutes FAISS Store

| File | Contents |
|---|---|
| `manifest.json` | List of jurisdiction codes |
| `state_<code>_index.faiss` | Dense index (dim 384) |
| `state_<code>_docs.pkl` | `citation`, `section_title`, `full_text`, `state` |
| `state_<code>_bm25.pkl` | Pickled BM25 with tokenized corpus |

**Jurisdictions:** 51 total (50 states + `federal` + `dc` + `pr`).

**Load-time normalization:** Every FAISS index is reconstructed and L2-normalized in memory, then rebuilt as `IndexFlatIP`. This guarantees `IndexFlatIP` inner product == cosine similarity, regardless of how the store was originally built.

---

## 4. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Delta-only intake** | Prevents LLM drift and stale fact echoing |
| **Regex fallbacks** | Deterministic safety net for the two most critical structured facts |
| **RRF fusion** | Scale-free; robust to incomparable score distributions |
| **Soft scoring with anti-signals** | Suppresses unrelated code sections that vector search would otherwise return |
| **Code-level anti-signals (-5.0)** | Strongest signal of irrelevance is the code name itself |
| **Known-facts guard** | Prevents the LLM from re-asking for known facts |
| **Dynamic citation rule** | One static prompt cannot handle 4 distinct retrieval scenarios |
| **L2-normalize on load** | Guarantees cosine semantics regardless of store provenance |
| **4-bit NF4 quantization** | Fits 8B model in ~5 GB, leaving headroom on a T4 |
| **Greedy decoding** | Reproducible, deterministic output |
| **BGE query prefix** | Required for asymmetric search with bge-small |

---

## 5. Safety & Guardrails

| Guardrail | Implementation |
|---|---|
| No hallucinated statutes | Dynamic citation rule + candidate count signal |
| No inferred jurisdiction | Explicit intake prompt rule |
| No inferred dollar amounts | Regex + numeric parsing on validated deltas |
| No re-asking known facts | Known-facts guard block |
| No gender assumption | Explicit prompt rule |
| No unrequested attorney-letterhead advice | Explicit prompt rule |
| Illegal act refusal | Explicit prompt rule #9 |
| State-corpus mismatch handled | `_qa_state_ok` filter + `corpus_loaded` flag |
| Corrupt store detection | `assert ntotal == len(docs)` on load |
| Invalid delta fields dropped | `_validate_delta` whitelist |

---

## 6. Extending the System

### Add a new case type

1. Add a phrase to `_CASE_QUERY_TEMPLATES`.
2. Add positive terms to `_CASE_SIGNALS`.
3. Add negative terms to `_CASE_ANTI_SIGNALS`.
4. Add code names to `_CASE_CODE_ANTI`.

### Add a new jurisdiction

1. Build a new FAISS store for the jurisdiction.
2. Add the code to `manifest.json`.
3. Ensure `state_<code>_index.faiss`, `state_<code>_docs.pkl`, and `state_<code>_bm25.pkl` exist.

### Add a new intake field

1. Add key to `_ALLOWED_KEYS`.
2. Add normalization logic to `_validate_delta`.
3. Add an example to the intake prompt's few-shot block.
4. Add a line to the known-facts guard in `get_legal_triage`.
5. Add the default to `start_legal_chat`'s initial `intake_state`.

### Swap the LLM

Change `LLM_MODEL_ID` in `config.py`. Verify the tokenizer exposes `apply_chat_template`.

### Swap the embedding model

Change `EMBED_MODEL_ID`. If the new model does not require a query prefix, remove the BGE prefix in `_retrieve_statutes` and `get_legal_triage`.

---

## 7. Performance Notes

| Operation | Approx. cost (T4) |
|---|---|
| Model load (first turn only) | ~2 min |
| Q&A embedding + search | ~50 ms |
| Statute dense + BM25 + RRF + soft rank | ~150–300 ms |
| LLM intent classification | ~1 s |
| LLM query translation | ~2 s |
| LLM intake extraction | ~2 s |
| LLM final response (600 tokens) | ~15–30 s |

**Per-turn latency:** ~20–40 s on a T4.

**Bottleneck:** LLM generation. Consider smaller models, streaming, or quantized KV cache to reduce.

---

## 8. Function Index

| Function | Purpose |
|---|---|
| `load_resources` | Load FAISS + models into globals |
| `start_legal_chat` | Interactive chat loop |
| `get_legal_triage` | Retrieve + generate response |
| `classify_input_intent` | 4-way intent label |
| `translate_query` | Standalone query rewrite |
| `update_intake_state` | Delta-only fact extraction |
| `normalize_state` | Name → 2-letter code |
| `parse_amount` | String → positive float |
| `_validate_delta` | Whitelist + normalize delta |
| `_apply_fallbacks` | Regex state + amount fallback |
| `fallback_extract_state` | State regex |
| `fallback_extract_amount` | Amount regex |
| `_retrieve_statutes` | Hybrid statute search |
| `_rrf_fuse` | Reciprocal Rank Fusion |
| `_soft_score` | Case-type scoring |
| `_build_citation_rule` | Dynamic prompt fragment |
| `_normalize_index_in_memory` | FAISS re-normalization |
| `_generate` | LLM call wrapper |
| `_off_topic_response` | Canned off-topic / chit-chat reply |

---

*Save this as `docs/ARCHITECTURE.md` in the repo. It pairs with `README.md` (setup) and `PROJECT.md` (full documentation).*
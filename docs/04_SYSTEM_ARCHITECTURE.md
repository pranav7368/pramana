# 04 — System Architecture

**System:** PRAMANA — Pipeline for Retrieval-Augmented Multilingual Assurance via NLI-grounded Auto-correction
**Audience:** implementers and reviewers
**Design stance:** production-grade modularity — every stage is an interface with a swappable implementation

---

## 1. Architectural Principles

These five decisions constrain everything else in this document. Each is justified in the sections that follow.

| # | Principle | Rationale |
|---|---|---|
| **P1** | **Black-box over the generator** | The assurance layer never requires model internals or fine-tuning. It therefore wraps API-served models and open-weight models alike, and survives a generator swap. Contrast Self-RAG, which requires retraining. |
| **P2** | **Claim as the unit of truth** | Verdicts, evidence links, and corrections all operate on atomic claims. Answer-level scores cannot drive targeted correction. |
| **P3** | **Separation of measurement and action** | Detection produces verdicts; confidence fuses evidence; the policy engine decides. No stage both measures and acts. This makes each independently testable and independently ablatable. |
| **P4** | **Deterministic control flow** | The correction loop is a bounded state machine, not an autonomous agent. Latency and cost are provably bounded; behaviour is reproducible. |
| **P5** | **Language as configuration, not as branching code** | Language-specific behaviour lives in profiles and model registries. There is no `if language == "ta"` in business logic. Adding a fourth language is a config change. |

---

## 2. High-Level View

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             CLIENT (Web UI / API)                            │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │  POST /v1/ask  {query, language?}
┌───────────────────────────────────▼──────────────────────────────────────────┐
│                          API LAYER  (FastAPI)                                │
│  request validation · tracing · auth · rate limiting · response assembly     │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────────┐
│                        ORCHESTRATOR  (LangGraph state machine)               │
│                                                                              │
│   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐ │
│   │ Lang   │──▶│ Retrieve │──▶│ Generate │──▶│  Detect  │──▶│ Confidence │ │
│   │ Norm.  │   │          │   │          │   │          │   │            │ │
│   └────────┘   └────▲─────┘   └────▲─────┘   └──────────┘   └─────┬──────┘ │
│                     │              │                              │        │
│                     │              │         ┌────────────────────▼──────┐ │
│                     └──────────────┴─────────┤   CORRECTION POLICY      │ │
│                        re-retrieve / regenerate│  (bounded loop, k ≤ 2)  │ │
│                                                └────────────┬─────────────┘ │
└─────────────────────────────────────────────────────────────┼───────────────┘
                                                              │
                                    ┌─────────────────────────▼───────────────┐
                                    │  RESPONSE                               │
                                    │  answer · confidence · claim verdicts   │
                                    │  · evidence citations · action taken    │
                                    │  · trace                                │
                                    └─────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  SUPPORTING SERVICES                                                         │
│  Model Registry · Config Profiles · FAISS Index · Cache · Trace Store        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Data Model

Every inter-stage contract is a typed object. This is what makes the modules independently testable.

```python
# src/pramana/schemas.py  (design specification)

Language = Literal["en", "hi", "ta"]
Script   = Literal["native", "roman"]

class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    language: Language
    metadata: dict          # source, section, page, effective_date

class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    sparse_score: float
    rerank_score: float | None
    rank: int

class RetrievalResult:
    query: str
    normalized_query: str
    language: Language
    script: Script
    chunks: list[RetrievedChunk]
    # --- confidence signal S1 ---
    top_score: float
    score_margin: float      # score[0] - score[1]
    mean_top_k_score: float

class Draft:
    text: str
    # --- confidence signal S3 (None for API models without logprobs) ---
    mean_token_logprob: float | None
    predictive_entropy: float | None
    samples: list[str]       # --- signal S4: self-consistency resamples ---

class Claim:
    claim_id: str
    text: str                # atomic, self-contained
    source_span: tuple[int, int]   # char offsets into Draft.text
    language: Language

Verdict = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]

class ClaimVerdict:
    claim: Claim
    verdict: Verdict
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    margin: float                    # --- signal S2 ---
    supporting_chunk_ids: list[str]  # evidence attribution
    verification_language: Language  # which arm/language verification ran in

class ConfidenceReport:
    score: float                     # calibrated, [0, 1]
    raw_score: float                 # pre-calibration
    signals: dict[str, float]        # S1..S4 feature values
    calibrator: str                  # "temperature" | "isotonic" | "none"
    band: Literal["HIGH", "MEDIUM", "LOW"]

Action = Literal["ACCEPT", "REGENERATE", "RE_RETRIEVE", "PRUNE", "ABSTAIN"]

class AssuranceResult:
    final_answer: str
    confidence: ConfidenceReport
    claim_verdicts: list[ClaimVerdict]
    action_history: list[Action]
    iterations: int
    regressed: bool                  # revision rejected and rolled back
    latency_ms: dict[str, float]     # per stage
    trace_id: str
```

---

## 4. Module Specifications

### 4.1 Stage 0 — Language & Script Normalisation
`src/pramana/ingestion/language.py`

| Aspect | Specification |
|---|---|
| **Input** | Raw user query string |
| **Output** | `(normalized_query, Language, Script)` |
| **Responsibilities** | Language identification; script detection (Devanagari / Tamil / Latin); optional transliteration of Romanised input to native script; Unicode NFC normalisation; whitespace and punctuation cleanup |
| **Why it exists** | The Script Gap finding (arXiv:2512.10780) shows the same language in a different script produces different model behaviour. Real enterprise users type Romanised Hindi constantly. Ignoring this is the difference between an academic prototype and a deployable system. |
| **Failure mode** | Code-mixed queries. Handled by segment-level language tagging with a dominant-language fallback. |

### 4.2 Stage 1 — Retrieval
`src/pramana/retrieval/`

**Hybrid design:**

```
query ──┬──▶ dense encoder (BGE-M3) ──▶ FAISS ──▶ top-N dense
        │                                              │
        └──▶ BM25 ──────────────────────▶ top-N sparse │
                                                       ▼
                                       Reciprocal Rank Fusion
                                                       │
                                       cross-encoder rerank
                                                       │
                                                    top-k C
```

| Decision | Justification |
|---|---|
| Hybrid rather than dense-only | Dense multilingual embeddings degrade on domain-specific enterprise terminology (documented in the cross-lingual RAG literature). BM25 catches exact product names, policy codes, and identifiers that embeddings blur. |
| Reciprocal Rank Fusion | Score-scale-free; no per-language weight tuning required |
| Cross-encoder rerank | Highest-leverage precision improvement; small model, acceptable latency |
| Per-language index | Avoids cross-language score contamination; also enables clean per-language retrieval metrics |

**Emits confidence signal S1:** `top_score`, `score_margin`, `mean_top_k_score`.

### 4.3 Stage 2 — Generation
`src/pramana/generation/`

| Aspect | Specification |
|---|---|
| **Interface** | `Generator.generate(query, context, language) -> Draft` |
| **Implementations** | `HFLocalGenerator` (Qwen / Llama-3 / Gemma via transformers), `ApiGenerator` |
| **Prompting** | Strict grounding instruction; answer must be in the query language; explicit permission to say "the documents do not contain this" |
| **Sampling** | One greedy draft + *n* stochastic samples (default *n* = 4) for signal S4 |
| **Emits** | Signal S3 (mean token log-probability, predictive entropy) where the backend exposes log-probabilities |

> **Design note.** The generation prompt permits abstention. Many RAG systems forbid "I don't know",
> which manufactures hallucination. A system that can never abstain must fabricate when evidence is
> absent.

### 4.4 Stage 3 — Detection ★
`src/pramana/detection/`

Two sub-components.

**(a) Claim Decomposer** — `decomposer.py`

| Aspect | Specification |
|---|---|
| **Contract** | `decompose(draft: Draft, language: Language) -> list[Claim]` |
| **Requirements** | Claims must be *atomic* (one proposition), *self-contained* (no unresolved pronouns), *span-linked* (char offsets into the draft, so corrections can be applied surgically) |
| **Implementation** | LLM-based decomposition with a few-shot prompt per language, plus a rule-based sentence-splitter fallback |
| **Indic-specific concern** | Hindi and Tamil are pro-drop languages; the decomposer must restore elided subjects or the resulting claim is unverifiable for the wrong reason. Tamil agglutination also means a single orthographic word may carry multiple propositions. |
| **Ablation hook** | Granularity is a tunable parameter — the literature reports that optimal atomicity is verifier-dependent |

**(b) NLI Grounding Verifier** — `verifier.py`

| Aspect | Specification |
|---|---|
| **Contract** | `verify(claims, context, language, arm) -> list[ClaimVerdict]` |
| **Core operation** | For each claim: premise = retrieved context, hypothesis = claim, classify entailment / contradiction / neutral |
| **Evidence attribution** | Verified per-chunk as well as against the concatenated context, so each SUPPORTED claim links to specific `chunk_id`s — this is what makes citations trustworthy |
| **Batching** | All (claim × chunk) pairs batched in a single forward pass; this is the primary latency optimisation |
| **Arms** | `native` · `translate` · `hybrid` (see §5) |
| **Emits** | Signal S2 (verdict distribution, probability margins) |

**Thresholding.** Rather than taking argmax, per-language thresholds τ_entail and τ_contra are tuned on
the development set, because multilingual NLI models are not equally calibrated across languages. This
is a small detail with a large effect on Tamil F1, and it is a legitimate finding to report.

### 4.5 Stage 4 — Confidence ★
`src/pramana/confidence/`

```
S1 retrieval    ┐
S2 entailment   ├──▶ feature vector ──▶ fusion model ──▶ raw score ──▶ calibrator ──▶ ĉ
S3 uncertainty  │      (12–16 dims)     (LogReg / GBM)                 (temp. scaling)
S4 consistency  ┘
```

| Aspect | Specification |
|---|---|
| **Training target** | Binary — is the answer fully faithful according to human annotation? |
| **Fusion model** | Logistic regression (primary) or gradient boosting (comparator). Chosen for interpretability: feature weights are directly reportable, and a compliance reviewer can be shown why a score was low. |
| **Calibration** | Temperature scaling; isotonic regression as ablation |
| **Per-language** | Calibrators fitted per language; cross-language transfer measured explicitly (**RQ2**) |
| **Degradation** | If S3 is unavailable (API model without log-probabilities), the model is refitted on the remaining features rather than imputing — imputation would silently corrupt calibration |
| **Output banding** | HIGH ≥ 0.75 · MEDIUM 0.45–0.75 · LOW < 0.45 (thresholds tuned on dev set, not fixed by intuition) |

### 4.6 Stage 5 — Correction ★
`src/pramana/correction/`

**State machine:**

```
                       ┌─────────────┐
                       │   ASSESS    │◀────────────────┐
                       └──────┬──────┘                 │
          ┌───────────┬───────┼────────┬──────────┐    │
          ▼           ▼       ▼        ▼          ▼    │
      ┌────────┐ ┌────────┐ ┌──────┐ ┌───────┐ ┌─────────┐
      │ ACCEPT │ │REGENER-│ │ RE-  │ │ PRUNE │ │ ABSTAIN │
      │        │ │  ATE   │ │RETRV │ │       │ │         │
      └───┬────┘ └───┬────┘ └──┬───┘ └───┬───┘ └────┬────┘
          │          │         │         │          │
          │          └────┬────┘         │          │
          │               ▼              │          │
          │        ┌─────────────┐       │          │
          │        │ RE-VERIFY   │───────┘          │
          │        └──────┬──────┘                  │
          │               │ improved? ──no──▶ ROLLBACK
          │               │ yes                     │
          │               └─────────┐               │
          ▼                         ▼               ▼
       ┌──────────────────────────────────────────────┐
       │                   RETURN                     │
       └──────────────────────────────────────────────┘
                    (loop depth ≤ k, default 2)
```

**Policy table:**

| Condition | Action | Implementation |
|---|---|---|
| No CONTRADICTED, no UNVERIFIABLE, ĉ = HIGH | ACCEPT | Return with citations |
| ≥ 1 CONTRADICTED, S1 healthy | REGENERATE | Re-prompt with flagged claims + contradicting evidence surfaced explicitly, under a hard grounding constraint |
| UNVERIFIABLE dominant **and** S1 weak | RE_RETRIEVE | Reformulate query using unverified claim content; retrieve; regenerate |
| Minority UNVERIFIABLE, rest SUPPORTED | PRUNE | Delete claim spans, re-render for coherence |
| ĉ = LOW after *k* iterations, or no supported claims | ABSTAIN | Explicit insufficient-evidence response; flag for human review |

**Guard rails (P4):**
- Bounded loop depth, default *k* = 2, hard latency budget enforced
- **Rollback**: a revision is accepted only if its claim-verdict profile strictly improves — otherwise
  the previous answer is restored and `regressed = True` is recorded
- No blanket "check your work" self-critique; correction addresses flagged claims only
- Every action is logged to the trace for auditability

---

## 5. Cross-Lingual Verification Arms

The experimental variable for RQ4, implemented as a strategy interface.

```
ARM A — NATIVE
  query(ta) → retrieve(ta) → generate(ta) → decompose(ta) → NLI_multilingual(ta) → verdicts

ARM B — TRANSLATE-THEN-VERIFY
  query(ta) → retrieve(ta) → generate(ta) → decompose(ta)
                                              ↓ MT
                          claims(en) + context(en) → NLI_english → verdicts → map back to ta spans

ARM C — HYBRID (post-retrieval translation)
  query(ta) → retrieve(ta) → context(ta) ─MT→ context(en)
                                              ↓
                        generate with both → decompose → NLI_english → verdicts → map to ta spans
```

| Arm | Strength | Weakness |
|---|---|---|
| A | No translation loss; preserves culturally specific terms | Depends on multilingual NLI quality, weakest for Tamil |
| B | Uses the strongest available NLI models | Translation error enters twice; span mapping back to the source is lossy |
| C | Reported in the literature to improve both reasoning and surface fidelity | Highest cost; most complex |

**Span mapping** is the non-obvious engineering problem in arms B and C: a verdict computed on an
English claim must be attributed back to a Tamil character span so that PRUNE can operate. Implemented
via alignment on decomposition indices rather than token alignment — claims are tracked by ID through
translation, avoiding word-level alignment entirely.

---

## 6. Configuration & Profiles

Per principle P5, all language- and hardware-specific behaviour is configuration.

```yaml
# src/pramana/config/profiles/colab-t4.yaml
profile: colab-t4

generator:
  backend: hf_local
  model: Qwen/Qwen2.5-7B-Instruct
  quantization: 4bit
  n_consistency_samples: 4

embeddings:
  model: BAAI/bge-m3
  batch_size: 16

reranker:
  model: BAAI/bge-reranker-v2-m3
  top_k: 5

nli:
  arm: native
  model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
  thresholds:
    en: {entail: 0.60, contra: 0.55}
    hi: {entail: 0.55, contra: 0.50}
    ta: {entail: 0.50, contra: 0.50}   # tuned on dev set

confidence:
  fusion: logistic_regression
  calibrator: temperature
  bands: {high: 0.75, medium: 0.45}

correction:
  max_iterations: 2
  latency_budget_ms: 12000
  enable_actions: [ACCEPT, REGENERATE, RE_RETRIEVE, PRUNE, ABSTAIN]
```

Three shipped profiles: `laptop-dev`, `colab-t4`, `api`. Model names above are **candidates to be
validated during implementation**, not final commitments — availability changes.

### 6.1 Hardware Reality and the Split-Execution Strategy

The development machine for this project has been profiled and imposes a hard architectural constraint:

| Resource | Measured | Consequence |
|---|---|---|
| GPU | Intel Iris Xe (integrated), no CUDA | Local LLM inference is not viable |
| CPU | Intel i5-1135G7, 4 cores / 8 threads | Encoder-sized models only |
| RAM | 7.7 GB total | **The binding constraint** — a 7B generator cannot be resident |
| Disk | ~49 GB free | Sufficient for model cache and indices |

This is not a limitation to be worked around quietly; it **drives a design decision** that improves the
system. Because generation must run off-machine, the generator is forced to sit behind a clean interface
— which is exactly the black-box property principle **P1** demands. A constraint and a design goal
coincide.

**Split execution — generation is remote, everything else is local:**

```
┌────────────────────────────────────────┐      ┌──────────────────────────┐
│  LAPTOP  (all code runs here)          │      │  FREE HOSTED LLM APIs    │
│                                        │      │  Groq · Google AI Studio │
│  • Retrieval (FAISS + BM25)            │      │  · OpenRouter · Cerebras │
│  • NLI verifier (CPU)                  │ HTTP │                          │
│  • Confidence fusion + calibration     │ ────▶│  Llama-3 · Gemma · Qwen  │
│  • Correction policy engine            │      │                          │
│  • StubGenerator (dev, no model)       │ ◀────│  → drafts + samples      │
│  • Evaluation, analysis, figures       │ JSON │                          │
│  ~50 MB RAM for the API client         │      │  (Colab T4 = fallback)   │
└────────────────────────────────────────┘      └──────────────────────────┘
                    │
                    ▼
        data/cache/generations/   ← content-addressed cache
        data/processed/drafts_{lang}.jsonl
```

**No GPU is used anywhere.** The generation script is pure HTTP and runs on the laptop itself; only the
*inference* happens remotely, on someone else's hardware, at no cost.

**The mechanism that makes this work: generation is decoupled and cached.**

1. Run generation once for the whole evaluation set against free hosted APIs; write every `Draft`
   (including self-consistency samples, and log-probabilities where the provider supplies them) to
   `data/processed/drafts_{lang}.jsonl`.
2. Every subsequent experiment — detection, confidence, correction, all ablations — replays from the
   cached drafts. **No GPU is required at any point in this project.**

This is why the split works rather than merely being tolerable: stages 3–5 are the contribution, and
they are encoder- and CPU-bound, not generator-bound.

**Exception — the correction loop.** REGENERATE and RE_RETRIEVE need fresh generation, so they cannot
be fully replayed from a static cache. Handled by pre-computing correction-round drafts in the same
batch pass, keyed by `(query_id, round, action)`, so the laptop can replay bounded correction loops
offline.

**Revised profiles:**

| Profile | Generator | Embeddings | NLI | Use |
|---|---|---|---|---|
| `laptop-dev` | `StubGenerator` (fixtures, no model) | MiniLM multilingual (~470 MB) | mDeBERTa-v3-base XNLI | Development, unit tests, fast iteration |
| `api` | **Free hosted APIs — primary path** (Groq / Google AI Studio / OpenRouter) | MiniLM or BGE-M3 | mDeBERTa | **Draft generation and all main runs** |
| `replay` | Cached drafts from disk | MiniLM | mDeBERTa | All experiments and ablations after generation |
| `colab-t4` | Qwen 2.5 7B (4-bit) on Colab GPU | BGE-M3 | mDeBERTa | **Fallback only** — if API quotas prove insufficient |

**Generator abstraction (required by P1, made mandatory by free-tier limits).** All backends implement
one `Generator` protocol behind a **rate-limit-aware router** with **content-addressed disk caching**
keyed on `sha256(model, prompt, temperature, seed, n)`. The router round-robins across providers, backs
off on HTTP 429, and fails over when a daily quota is exhausted, so an overnight run survives a single
provider running dry.

**Signal S3 under hosted inference.** Many free endpoints do not return token log-probabilities. Per
§4.5, the fusion model is then **refitted without S3** rather than imputing it. A substitute is
available at no extra cost: semantic dispersion across the self-consistency samples (S4), which are
generated regardless. `scripts/probe_providers.py` measures per-provider logprob support empirically
rather than trusting documentation.

**Memory discipline for `laptop-dev` / `replay`** (with ~6 GB usable, models load one at a time):
lazy loading, explicit unload between stages, NLI `batch_size` capped at 8, embeddings pre-computed
once and memory-mapped from disk rather than held resident.

**Resumability** applies to both API and Colab paths: batch jobs write JSONL incrementally with a flush
per item and skip already-completed `query_id`s on restart. A run that dies at 70% must resume, not
restart — a requirement under API rate limits just as much as under Colab disconnects.

**Reproducibility caveat specific to hosted models.** A hosted model can be updated or deprecated
without notice, so the endpoint is not a stable artefact. **The cached drafts are the reproducible
artefact**, and `drafts_*.jsonl` is archived alongside the report with the exact model string and
generation date recorded in the run manifest.

---

## 7. API Contract

```http
POST /v1/ask
{
  "query": "Mera claim reject kyun hua?",
  "language": null,            // auto-detect if omitted
  "assurance": true,
  "max_corrections": 2
}
```

```jsonc
200 OK
{
  "answer": "...",
  "confidence": {
    "score": 0.82,
    "band": "HIGH",
    "signals": { "retrieval_margin": 0.31, "entail_ratio": 1.0,
                 "mean_logprob": -0.42, "self_consistency": 0.95 }
  },
  "claims": [
    { "text": "...", "verdict": "SUPPORTED",
      "evidence": ["doc_14#c3"], "entailment_prob": 0.91 }
  ],
  "action_history": ["REGENERATE", "ACCEPT"],
  "iterations": 1,
  "regressed": false,
  "detected_language": "hi",
  "detected_script": "roman",
  "latency_ms": { "retrieval": 84, "generation": 1420,
                  "detection": 610, "confidence": 12, "correction": 1380 },
  "trace_id": "..."
}
```

Additional endpoints: `GET /v1/health`, `POST /v1/verify` (verification only, no generation — useful
for auditing an existing system's logs), `GET /v1/trace/{trace_id}`.

---

## 8. Evaluation Harness

`src/pramana/evaluation/`

```
scripts/run_experiment.py --config experiments/main.yaml
        │
        ├─▶ load eval set (en/hi/ta)
        ├─▶ for each system in [vanilla, ragas_thresh, llm_judge,
        │                       sentence_nli, self_consistency, PRAMANA]:
        │       for each language:
        │           run pipeline → collect AssuranceResult
        ├─▶ compute metrics (retrieval, detection, calibration,
        │                    correction, efficiency)
        ├─▶ paired bootstrap significance tests
        └─▶ emit reports/{run_id}/{tables.csv, figures/, raw.jsonl}
```

Every run writes a manifest recording config hash, model revisions, dataset version, and random seeds.
Reproducibility is a stated deliverable, not an aspiration.

---

## 9. Testing Strategy

| Level | Coverage |
|---|---|
| **Unit** | Language ID and transliteration; RRF fusion; claim decomposition invariants (atomicity, span validity); NLI thresholding; policy-table branch selection; calibration maths |
| **Contract** | Every stage validated against its typed schema; fixture-driven, no model calls |
| **Integration** | End-to-end on a tiny fixture corpus with a stub generator — fast, deterministic, CI-friendly |
| **Regression** | Golden-file tests on a frozen sample: verdicts and actions must not drift silently |
| **Evaluation** | The full harness (§8) — treated as an experiment, not a test |

The stub generator matters: it lets the entire pipeline be tested without a GPU, which is what makes
this project tractable on a laptop.

---

## 10. Deployment View

```
┌────────────┐    ┌──────────────┐    ┌───────────────────┐
│  Web UI    │───▶│  FastAPI     │───▶│  PRAMANA core     │
│ (demo)     │    │  (uvicorn)   │    │  (in-process)     │
└────────────┘    └──────┬───────┘    └─────────┬─────────┘
                         │                      │
                         ▼                      ▼
                  ┌─────────────┐      ┌──────────────────┐
                  │ Trace store │      │ FAISS indices    │
                  │ (JSONL)     │      │ (per language)   │
                  └─────────────┘      └──────────────────┘
                                                │
                                       ┌────────▼─────────┐
                                       │ Model cache      │
                                       │ (HF local / API) │
                                       └──────────────────┘
```

Single-process deployment is deliberate for this project's scale. Documented scale-out path: extract the
NLI verifier into its own service (it is the batchable, GPU-bound component), keep orchestration
stateless, move the trace store to a database.

---

## 11. Traceability — Architecture to Objectives

| Module | Objective | Research Question | Contribution |
|---|---|---|---|
| Language normalisation | O1 | RQ4 | — |
| Hybrid retrieval | O1 | — | — |
| Generation | O1 | — | — |
| Claim decomposer | O3 | RQ1 | C1 |
| NLI verifier | O3, O4 | RQ1, RQ4 | C1, C2 |
| Confidence fusion + calibration | O5 | RQ2 | C3 |
| Correction policy engine | O6 | RQ3 | C1, C5 |
| Evaluation harness | O7 | all | C4, C5 |
| API + demo | O8 | — | C1 |

Every module traces to an objective. Nothing is built that no objective requires — which is also the
answer to *"why doesn't your system do X?"* in a design review.

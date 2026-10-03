# 07 — Implementation Guide

**Purpose:** exact setup and execution instructions for this project's actual hardware.
**Profiled machine (2026-08-10):** Intel i5-1135G7 · 4C/8T · **7.7 GB RAM** · Intel Iris Xe (no CUDA) ·
49.5 GB free disk · Python 3.12.4 · Git 2.39.0

---

## 1. The Constraint, and Why It Is Not Fatal

There is no CUDA GPU and only 7.7 GB of system RAM. A 7-billion-parameter generator cannot run here,
and neither can a 1.5B one at the throughput this evaluation requires (300+ questions × 3 languages ×
6 systems × correction rounds).

**This does not block the project**, for one structural reason:

> The research contribution — claim decomposition, NLI verification, confidence fusion, correction
> policy — is **encoder-bound and CPU-feasible**. Only *generation* needs a GPU, and generation is the
> part that is **standard, not novel**.

So generation is treated as a **batch pre-computation step**, executed against **free hosted LLM APIs**
(§4), cached to disk, and replayed locally forever after. Everything that constitutes the contribution
then runs on the laptop, offline, repeatedly, at zero cost.

```
   ONE-TIME, VIA FREE API                REPEATED, ON LAPTOP (CPU)
   ──────────────────────                ─────────────────────────
   generate drafts for all queries       detection    ┐
   + n self-consistency samples          confidence   ├── all experiments
   + correction-round drafts             correction   │   all ablations
   + token log-probs (if available)      evaluation   ┘   all analysis
              │                                ▲
              └──── drafts_{lang}.jsonl ───────┘
                    (a few MB of text)

   The generation script also runs ON the laptop — it is pure HTTP,
   ~50 MB of RAM. No GPU is involved anywhere in this project.
```

**Practical consequence:** no GPU is needed at any point. A script runs overnight against free API
quotas, and once the drafts are cached, every subsequent experiment is local, free, and repeatable.
Google Colab is retained only as a fallback (§5).

---

## 2. Memory Budget

With 7.7 GB total, roughly **5.5–6 GB is usable** after Windows. Every model choice below is made
against that number.

| Component | Model | Disk | Resident RAM | Verdict |
|---|---|---|---|---|
| Embeddings (laptop) | `paraphrase-multilingual-MiniLM-L12-v2` | ~470 MB | ~0.6 GB | ✅ Comfortable |
| Embeddings (upgrade) | `BAAI/bge-m3` | ~2.3 GB | ~2.5 GB | ⚠️ Only with ≥ 5 GB free; else keep MiniLM |
| NLI verifier | `mDeBERTa-v3-base-xnli` | ~1.1 GB | ~1.4 GB | ✅ Works, slow |
| Reranker (optional) | `bge-reranker-v2-m3` | ~2.3 GB | ~2.5 GB | ⚠️ Tight — treat as optional |
| FAISS index | ~5k chunks × 384 dims | ~8 MB | ~0.05 GB | ✅ Trivial |
| **Generator (API client)** | HTTP only — no weights | 0 | **~50 MB** | ✅ **This is why the API path solves the problem** |
| Generator (local, for reference) | Qwen 2.5 7B (4-bit) | ~5 GB | ~6 GB | ❌ **Not on this laptop** |

**Discipline rules enforced in code:**
1. **One model resident at a time.** Load → use → `del` → `gc.collect()`. The pipeline is staged, so
   this costs load time, not correctness.
2. **Embeddings computed once**, persisted to disk, memory-mapped thereafter. Never re-embed.
3. **NLI batch size ≤ 8** on laptop. Larger batches will trigger swap and slow things by 10×.
4. **Close Chrome before experiment runs.** At profiling time only 0.8 GB was free — that alone would
   fail. Target at least 4 GB free before a run.

Check free memory before any run:

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object @{n='FreeGB';e={[math]::Round($_.FreePhysicalMemory/1MB,2)}}
```

---

## 3. Laptop Setup

### 3.1 Environment

```powershell
cd "$env:USERPROFILE\Desktop\CaseStudy"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

> If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3.2 Install — CPU-only PyTorch

**The CPU-only wheel is mandatory.** The default `pip install torch` pulls ~2.5 GB of CUDA libraries
that are useless on Iris Xe and will waste a significant fraction of your free disk.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-laptop.txt
```

### 3.3 `requirements-laptop.txt`

```
# --- core ---
numpy>=1.26,<2.2
pydantic>=2.7
pyyaml>=6.0
python-dotenv>=1.0

# --- retrieval ---
faiss-cpu>=1.8
rank-bm25>=0.2.2
sentence-transformers>=3.0

# --- NLI / models ---
transformers>=4.44
sentencepiece>=0.2
protobuf>=4.25

# --- multilingual text handling ---
indic-transliteration>=2.3    # Devanagari <-> Roman
langdetect>=1.0.9
regex>=2024.5

# --- confidence / calibration ---
scikit-learn>=1.5
scipy>=1.13

# --- orchestration ---
langchain>=0.2
langchain-community>=0.2
langgraph>=0.1

# --- evaluation ---
ragas>=0.1.9
datasets>=2.20
pandas>=2.2
matplotlib>=3.9

# --- serving ---
fastapi>=0.111
uvicorn[standard]>=0.30

# --- dev ---
pytest>=8.2
pytest-cov>=5.0
ruff>=0.5
rich>=13.7
tqdm>=4.66
```

> **Note on RAGAS.** RAGAS expects an LLM judge. On the laptop it is used only in *replay* mode against
> cached judgements, or run on Colab. Do not attempt live RAGAS evaluation locally.

### 3.4 Environment file

```ini
# .env  (never commit this — it is gitignored)
PRAMANA_PROFILE=laptop-dev
HF_HOME=D:/hf_cache            # relocate if C: gets tight
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=4
```

`OMP_NUM_THREADS=4` matches physical cores; leaving it unset lets PyTorch oversubscribe 8 logical
threads and run *slower* on this CPU.

### 3.5 Verify

```powershell
python scripts/check_env.py
```

Expected: Python ≥ 3.11, torch CPU build, FAISS import, ≥ 4 GB free RAM, ≥ 15 GB free disk, and a
successful tiny embedding round-trip.

---

## 4. Generation Backend — Free APIs (Recommended Primary Path)

Free hosted inference removes the GPU problem entirely and is **preferred over Colab** for this project.
Colab is retained as a fallback (§5).

### 4.1 Why API beats Colab here

| | Free API | Colab free T4 |
|---|---|---|
| Session timeouts | None | Disconnects, daily quota |
| Laptop RAM used | ~50 MB (just HTTP) | N/A but needs browser open |
| Unattended overnight runs | ✅ Yes | ❌ Dies when session drops |
| Model variety for ablation | Qwen + Llama-3 + Gemma from one interface | One model per session load |
| Setup cost | API key, 5 minutes | Notebook, mounting Drive, reinstalls each session |
| Reproducibility risk | Endpoint may change under you | Model file is pinned |

The decisive point: **generation is already a cached batch step** in this architecture (§1). An API fits
that shape perfectly — a script runs overnight on the laptop, writes JSONL, and stops. Nothing is
resident in memory.

### 4.2 Provider Landscape

> ⚠️ **Verify these before relying on them.** Free tiers change frequently, and the figures below come
> from comparison articles rather than first-hand testing. Run `scripts/probe_providers.py` (§4.5) to
> measure the actual limits on your own key before planning around any number here.

| Provider | Reported free limits | Relevant models | Card needed | Fit for this project |
|---|---|---|---|---|
| **Groq** | ~30 RPM · ~1,000 req/day | Llama 3.3 70B, Gemma, Mixtral | No | ★★★ **Primary.** Fast, generous daily cap, has two of your three named model families |
| **Google AI Studio** | ~5–15 RPM · ~20–1,500 req/day | Gemini Flash, Gemma variants | No | ★★★ **Strong for Indic.** Google's models tend to be well-resourced for Hindi/Tamil |
| **Cerebras** | ~30 RPM · ~1M tokens/day | Llama 3.3 70B | No | ★★ Good volume, narrower model choice |
| **OpenRouter** | ~20 RPM · 50 req/day (1,000 after a one-time $10 top-up) | Qwen3, Llama 3.3, Gemma 3, DeepSeek — ~30 free models | No | ★★★ **Best model coverage** — all three named families through one OpenAI-compatible interface. 50/day is too low without the top-up. |
| **Mistral** | ~1B tokens/month | Mistral Small/Large | No | ★ Huge volume but models aren't in your problem statement |
| **GitHub Models** | ~15 RPM · 150–1,000 req/day | Llama, Phi, others | No | ★★ Useful as a spare |
| **HuggingFace Inference** | Community rate-limited | Many open models | No | ★ Unpredictable for batch work |

**Recommended combination:** Groq as primary + Google AI Studio as secondary + OpenRouter for models the
first two lack. Three keys, one router, roughly 2,500–3,500 requests/day combined.

### 4.3 Volume Budget — Does It Actually Fit?

| Item | Count |
|---|---|
| QA pairs per language | 300 |
| Languages | 3 |
| Total queries | **900** |
| Calls per query (1 greedy + 4 self-consistency samples) | 5 |
| Baseline generation calls | **4,500** |
| Correction rounds (~30% of queries trigger, ≤ 2 rounds) | ~700 |
| **Total, one generator** | **≈ 5,200 calls** |
| Full generator ablation (Qwen + Llama-3 + Gemma) | ≈ 15,600 calls |

| Plan | Daily capacity | Main run (5,200) | Full ablation (15,600) |
|---|---|---|---|
| Groq only | ~1,000/day | ~6 days | ~16 days |
| Groq + Gemini | ~2,500/day | ~2 days | ~7 days |
| Groq + Gemini + OpenRouter (topped up) | ~3,500/day | **~2 days** | **~5 days** |

**It fits comfortably** on a 12-week timeline, provided generation starts early and runs in the
background while you build the detection modules. This is the scheduling reason to build the API layer
in week 3, not week 8.

**Levers if you fall behind:** reduce self-consistency samples from 4 to 3 (−20%), reduce the eval set
to 200 per language for the ablation arms only (−33%), or run the generator ablation on a 100-query
subset rather than the full set. Each is a defensible scoping decision; record whichever you use.

### 4.4 Required Engineering — Three Non-Negotiables

**(a) Provider-agnostic adapter.** One interface, many backends. Principle P1 already demands the
generator be swappable; free-tier juggling makes it mandatory.

```python
class Generator(Protocol):
    def generate(self, query, context, language, n_samples=1) -> Draft: ...

# implementations: GroqGenerator, GeminiGenerator, OpenRouterGenerator,
#                  StubGenerator, CachedGenerator
```

**(b) Content-addressed disk cache.** Key on `sha256(model, prompt, temperature, seed, n)`. A repeated
request is served from disk, never re-billed against your quota.

> This is what makes the whole plan work. A crashed run resumes for free. A re-run of an experiment
> costs nothing. Development iteration costs nothing.

**(c) Rate-limit-aware router.** Round-robin across providers, exponential backoff on HTTP 429, and
**automatic failover** when a provider's daily quota is exhausted. The run continues on the next
provider rather than dying at 2 a.m.

```
request ──▶ cache? ──yes──▶ return
              │no
              ▼
         router: pick provider with remaining quota
              │
              ├─ 429? ──▶ backoff, then next provider
              ├─ quota exhausted? ──▶ next provider
              └─ success ──▶ write cache, write JSONL, flush
```

### 4.5 The Logprobs Question (Signal S3) — **RESOLVED BY MEASUREMENT**

> ### ✅ Measured result — Groq, 2026-08-10
>
> `scripts/probe_providers.py --provider groq` against a live key:
>
> | Property | Result |
> |---|---|
> | Reachable | ✅ 15 models listed, 162 ms first-response latency |
> | Model | `llama-3.3-70b-versatile` |
> | System role | ✅ Honoured |
> | Seed determinism | ✅ Reproducible — same seed, same output |
> | **Log-probabilities (S3)** | ❌ **Not returned** |
> | **Server-side `n>1`** | ❌ **Not supported** — adapter loops client-side |
> | Hindi (Devanagari) | ✅ `नमस्ते` echoed correctly |
> | Tamil | ✅ `வணக்கம்` echoed correctly |
>
> Groq's registry entry declared `multi_sample` and `json_mode`; **neither survived probing.** This is
> exactly why capabilities are measured rather than trusted — planning signal S3 around a documented
> flag would have failed silently in week 8.

**Consequences, now settled rather than speculative:**

1. **Signal S3 is unavailable on Groq.** The fusion model is fitted on **S1, S2 and S4**, with
   uncertainty estimated as **semantic dispersion across self-consistency samples** rather than token
   probabilities. This is a documented design decision, not a gap.
2. **Each self-consistency sample costs one request.** The §4.3 volume budget already assumed this
   (5 calls per query), so the estimate holds — but it means Groq alone runs the main job in ~5–6 days.
   A second key roughly halves that.
3. **Seed determinism works**, which is a genuine win for reproducibility: cached drafts can be
   regenerated exactly if the cache is ever lost.
4. **Indic script survives the round trip.** Note the limit of this evidence: the probe tested whether
   the model can *echo* a Hindi and a Tamil word, **not whether it generates good Hindi or Tamil**.
   Generation quality in those languages remains an open question, to be answered by the evaluation
   itself — do not cite the probe as evidence of multilingual competence.

**If a second provider is added**, re-run the probe. `router.guaranteed_capabilities()` returns the
intersection across live providers, which is the honest basis for deciding whether a signal can be
relied on for a whole run.

#### Second provider added — Google, 2026-08-10

| Property | Groq | Google |
|---|---|---|
| Model | `llama-3.3-70b-versatile` | `gemini-flash-lite-latest` |
| Latency | ~130–260 ms | ~670–820 ms |
| System role | ✅ | ✅ |
| Logprobs (S3) | ❌ | ❌ |
| Server-side `n>1` | ❌ | ❌ |
| **Seed determinism** | ❌ 2/3 distinct | ✅ 3/3 identical |
| Hindi / Tamil script | ✅ | ✅ |
| Free limits | ~30 rpm · ~1,000 rpd | ~15 rpm · ~1,500 rpd |

Combined capacity ≈ **2,500 requests/day**, which puts the ~5,200-call main run at **about two days**.

Two model-selection lessons, both learned the hard way:

- **Pinned Google model ids rot.** `gemini-2.0-flash` (the original default) and `gemini-2.5-flash`
  now return 404; `gemini-2.5-pro` returns free-tier `limit: 0`. The registry uses the floating
  `-latest` aliases instead. A twelve-week project cannot afford a default model that disappears.
- **Some models are reasoning models.** At `max_tokens=16`, `gemini-flash-latest` returned *empty*
  output and `gemma-4-31b-it` leaked its chain of thought into the answer. Give them a larger budget,
  or prefer the `-flash-lite-` variants.

#### ⚠️ Signal S4 requires a non-zero temperature

A live three-sample run at the default `temperature=0.0` returned three **identical** completions.
That is correct behaviour, and it makes self-consistency **degenerate** — the feature would be a
constant 1.0 for every query and carry no information.

Since S3 is unavailable, **S4 is now the only uncertainty signal**, so this matters more than it
otherwise would:

| Purpose | Temperature |
|---|---|
| The answer that is actually returned and verified | `0.0` — deterministic, reproducible |
| The self-consistency samples feeding S4 | `> 0` (start at `0.7`, tune on the dev set) |

The generation config must therefore carry **two** temperatures, not one. Sampling temperature becomes
an ablation variable: too low and S4 is constant, too high and sample divergence reflects noise rather
than uncertainty.

---

**Background — why this was expected.** Most free endpoints do not return token log-probabilities. One
analysis found only about 23% of reachable OpenRouter endpoints return logprobs when requested, and
provider support is inconsistent.

This affects **confidence signal S3** (generation uncertainty). Two honest responses:

1. **The architecture already handles it.** `04_SYSTEM_ARCHITECTURE.md` §4.5 specifies that when S3 is
   unavailable, the fusion model is **refitted on the remaining features** rather than imputing a value
   — imputation would silently corrupt calibration.
2. **A good substitute exists.** Uncertainty can be estimated from the **self-consistency samples you
   are already generating** (signal S4) by measuring semantic dispersion across samples rather than
   token probabilities. This needs no logprobs at all, and it makes S4 do double duty.

**Action:** `scripts/probe_providers.py` empirically tests each provider for logprobs support, actual
RPM/RPD, and multilingual output quality, and writes the results to
`reports/provider_capabilities.json`. Decide from measurement, not from documentation.

If no provider supplies logprobs, report it as a deliberate constraint: *"S3 was unavailable across all
free endpoints; uncertainty is estimated via semantic dispersion over self-consistency samples, and the
fusion model is fitted on S1, S2, S4."* That is a clean, defensible statement.

### 4.6 Caveats You Must Record in the Report

| Issue | Why it matters | Mitigation |
|---|---|---|
| **Model versions drift** | A hosted `llama-3.3-70b` may be silently updated; your results become unreproducible | Record the exact model string **and the generation date** in every run manifest. **The cached drafts are the reproducible artefact** — the endpoint is not. Archive `drafts_*.jsonl` alongside the report. |
| **Endpoints get deprecated** | The model you used may be gone when results are reviewed | Same mitigation — the cache is the record of what ran, and it can be replayed live in a demo |
| **Free tiers change** | Your plan may break mid-project | Multi-provider router; probe script re-run monthly |
| **Data leaves your machine** | Third parties receive your prompts | **Not an issue here — the corpus is synthetic and contains no real or personal enterprise data.** State this explicitly; it is one of the reasons the synthetic corpus was chosen (`03_PROPOSAL.md` §10). |
| **Non-determinism** | Even at temperature 0, hosted inference may vary slightly | Cache-first design means each experiment replays identical drafts; note it as a known limitation |

### 4.7 Setup

```ini
# .env  — NEVER commit this file
PRAMANA_PROFILE=api
GROQ_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
PRAMANA_CACHE_DIR=./data/cache/generations
```

```powershell
pip install groq google-generativeai openai httpx tenacity
python scripts/probe_providers.py          # measure real limits + logprobs support
python scripts/generate_drafts.py --lang en --limit 10   # smoke test
python scripts/generate_drafts.py --lang en              # full run, resumable
```

Confirm `.env` is gitignored before the first commit. An API key pushed to GitHub is detected and
abused within minutes.

---

## 5. Colab Workflow (Fallback)

Use Colab only if free-API quotas prove insufficient, or if you need a model no free endpoint serves.

### 5.1 What runs there

Only generation, and only in batch:

| Job | Output | Frequency |
|---|---|---|
| Baseline drafts + *n*=4 self-consistency samples + log-probs | `drafts_{lang}.jsonl` | Once per generator |
| Correction-round drafts, keyed `(query_id, round, action)` | `corrections_{lang}.jsonl` | Once per generator |
| BGE-M3 embeddings for the corpus | `embeddings_{lang}.npy` | Once |

Three generators (Qwen, Llama-3, Gemma) × 3 languages is the full sweep — one long session each,
checkpointed.

### 5.2 Checkpointing is mandatory

Free Colab disconnects. Every batch job must:

1. Write results **incrementally**, one JSON object per line, flushed after each item.
2. On start, read the existing output file and build a set of completed `query_id`s.
3. **Skip** completed items.
4. Mount Google Drive and write there, not to ephemeral local storage.

A run that dies at 70% must resume at 70%. This is a hard requirement in `scripts/colab_generate.py`,
not a nicety — without it a lost session costs a day.

### 5.3 Colab notebook skeleton

```python
# 1. Mount Drive
from google.colab import drive; drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/pramana/drafts_hi.jsonl'

# 2. Install
!pip install -q transformers accelerate bitsandbytes sentence-transformers

# 3. Load generator (4-bit)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb, device_map="auto")

# 4. Resume-aware loop
import json, os
done = set()
if os.path.exists(OUT):
    with open(OUT, encoding='utf-8') as f:
        done = {json.loads(l)['query_id'] for l in f if l.strip()}

with open(OUT, 'a', encoding='utf-8') as f:
    for item in queries:
        if item['query_id'] in done:
            continue
        draft = generate_with_samples(model, tok, item, n_samples=4)
        f.write(json.dumps(draft, ensure_ascii=False) + '\n')
        f.flush()          # ← survives a disconnect
```

### 5.4 Bringing results back

Download the JSONL files into `data/processed/`. They are **text only** — a few MB. Then every
experiment runs locally:

```powershell
python scripts/run_experiment.py --config experiments/main.yaml --replay data/processed/
```

---

## 6. Development Without Any Model — `StubGenerator`

The single most useful implementation decision for this hardware.

`StubGenerator` returns **hand-written fixture answers** containing deliberately planted hallucinations
of each type:

```python
class StubGenerator(Generator):
    """Deterministic fixture generator. No model, no GPU, no network.

    Fixtures carry planted hallucinations with known ground truth, so the
    detection and correction stages can be tested for correctness rather
    than merely for not crashing.
    """
    def generate(self, query, context, language) -> Draft:
        return self.fixtures[(query.id, language)]
```

Because the expected verdict for every planted claim is known in advance, the entire pipeline becomes
**unit-testable**:

| Fixture | Planted defect | Expected verdict | Expected action |
|---|---|---|---|
| `F01` | Number changed (30 → 15 days) | CONTRADICTED | REGENERATE |
| `F02` | Fact absent from context | UNVERIFIABLE | PRUNE |
| `F03` | Fully grounded | all SUPPORTED | ACCEPT |
| `F04` | Empty retrieval, fluent answer | all UNVERIFIABLE | ABSTAIN |
| `F05` | Negation dropped | CONTRADICTED | REGENERATE |
| `F06` | Hindi, pro-drop subject elided | decomposer must restore subject | — |
| `F07` | Tamil, agglutinated multi-proposition word | decomposer must split | — |

This runs in **under a second on CPU** and is what makes daily development possible on this machine.
It also gives the test suite genuine assertions instead of smoke tests.

---

## 7. Expected Runtimes on This Laptop

Rough, for planning. Measure and update after the first real run.

| Operation | Estimate |
|---|---|
| Embed 5,000 chunks (MiniLM, CPU) | 8–15 min (once) |
| FAISS index build | < 1 min |
| Retrieval, single query | < 100 ms |
| NLI verify one answer (~6 claims × 5 chunks = 30 pairs, batched) | 3–8 s |
| Full detection pass, 300 queries | 25–45 min |
| Confidence fusion + calibration fit | < 30 s |
| Full replay experiment, 1 language, 1 system | 30–60 min |
| Full sweep — 3 languages × 6 systems | **overnight** |

Long CPU runs are expected. Design for it: incremental writes, resumability, and `--limit N` for quick
iteration.

---

## 8. Build Order

Each step ends in something runnable. Nothing is built that cannot be immediately tested.

| Step | Build | Verifies |
|:--:|---|---|
| 1 | `schemas.py` + `check_env.py` | Environment sane, contracts fixed |
| 2 | Corpus loader, chunker, language/script normaliser | `pytest tests/test_ingestion.py` |
| 3 | FAISS + BM25 hybrid retrieval, English only | Recall@5 on a hand-made query set |
| 4 | `StubGenerator` + fixtures | Full pipeline runs end-to-end, no GPU |
| 5 | Claim decomposer (rule-based first, LLM later) | Atomicity and span-validity assertions |
| 6 | NLI verifier | F01–F07 fixtures produce expected verdicts |
| 7 | Confidence signals S1, S2 | Feature vectors on fixtures |
| 8 | Correction policy engine | Fixture → expected action |
| 9 | Colab draft generation | Real drafts cached to JSONL |
| 10 | Signals S3, S4 from cached drafts | Full feature set |
| 11 | Fusion model + calibration | ECE on dev set |
| 12 | Hindi + Tamil corpora | Per-language retrieval metrics |
| 13 | Cross-lingual arms A/B/C | RQ4 comparison |
| 14 | Evaluation harness + baselines | Results tables |
| 15 | FastAPI + demo UI | Service and evidence desk |

**Steps 1–8 need no GPU and no internet beyond the initial model download.** That is roughly half the
project, buildable entirely on this laptop.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MemoryError` / machine freezes | Model too large or batch too big | Close browsers; `batch_size=4`; confirm one model resident |
| Very slow NLI | Thread oversubscription | `OMP_NUM_THREADS=4`; `TOKENIZERS_PARALLELISM=false` |
| `pip install torch` pulls GBs | Default CUDA wheel | Use the `/whl/cpu` index URL (§3.2) |
| Disk filling up | HF cache in `C:` | Set `HF_HOME` to another drive |
| Colab session lost | Free-tier disconnect | Resume logic (§4.2) — verify it before a long run |
| Devanagari/Tamil renders as boxes | Terminal font | Cosmetic only; inspect in VS Code, not the console |
| RAGAS hangs locally | Trying to call an LLM judge | Run RAGAS on Colab or in replay mode |

---

## 10. Reproducibility

Every experiment writes a manifest:

```json
{
  "run_id": "2026-08-17T10-22-03",
  "profile": "laptop-dev",
  "config_sha256": "...",
  "dataset_version": "v0.2",
  "models": { "embed": "...", "nli": "...", "generator": "cached:drafts_v1" },
  "seeds": { "numpy": 42, "torch": 42 },
  "git_commit": "...",
  "host": { "cpu": "i5-1135G7", "ram_gb": 7.7, "gpu": "none" }
}
```

`generator: cached:drafts_v1` records exactly which cached draft set produced a result — without it,
split execution silently destroys reproducibility, because the same config could replay against
different drafts.

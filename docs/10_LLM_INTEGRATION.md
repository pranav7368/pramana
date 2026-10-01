# 10 — Connecting Any LLM

**Design goal:** the pipeline must never name a provider. Swapping Qwen for Llama-3, a hosted API for
a local server, or a real model for an offline stub is a **configuration change**, not a code change.

The architecture keeps the assurance layer
black-box over the generator. It is also what makes the generator ablation (**O7**) tractable and what
lets the project run on a laptop with no GPU.

---

## 1. The Contract

Everything reduces to one protocol:

```python
class LLMProvider(Protocol):
    name: str
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
    def capabilities(self, model: str) -> set[Capability]: ...
    def available_models(self) -> Iterable[str]: ...
    def health_check(self) -> bool: ...
```

Anything satisfying it plugs in. The pipeline imports **only** this protocol — never a vendor SDK.

```
       pipeline (detection · confidence · correction)
                          │
                          ▼
                    ┌───────────┐
                    │ LLMRouter │   cache · throttle · failover
                    └─────┬─────┘
        ┌─────────────────┼─────────────────┬──────────────┐
        ▼                 ▼                 ▼              ▼
  OpenAICompatible   GoogleGenAI      StubProvider    <your adapter>
        │
   ┌────┴─────────────────────────────────────────────┐
   ▼         ▼        ▼        ▼       ▼        ▼      ▼
 Groq  OpenRouter Cerebras Together Ollama  vLLM  LM Studio …
```

---

## 2. Three Ways to Connect

### 2.1 Already OpenAI-compatible → add YAML, write no code

Most of the ecosystem speaks the OpenAI Chat Completions dialect. Add an entry to
`src/pramana/config/providers.yaml`:

```yaml
  my_provider:
    adapter: openai_compatible
    base_url: https://api.example.com/v1
    api_key_env: MY_PROVIDER_KEY
    default_model: some-model-7b
    enabled: true
    priority: 4
    capabilities: [system_role, multi_sample]
    limits: { rpm: 60, rpd: 5000 }
    models:
      fast: some-model-7b
      big: some-model-70b
```

Put `MY_PROVIDER_KEY=...` in `.env` and it is live:

```powershell
python scripts/check_env.py          # confirms the key is seen
python scripts/probe_providers.py --provider my_provider
```

**Known to work through this adapter:** Groq, OpenRouter, Cerebras, Together, DeepInfra, DeepSeek,
Mistral, Fireworks, xAI, GitHub Models, Nebius — and every local server that emulates the API:
**Ollama, vLLM, LM Studio, llama.cpp, TGI, LocalAI**.

### 2.2 Local model → point at localhost

A local server needs no key. Entries for `ollama`, `lmstudio`, and `vllm` already ship, disabled:

```yaml
  ollama:
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    api_key_env: null          # no auth
    default_model: qwen2.5:1.5b
    enabled: true              # flip this
```

> **Worth knowing:** local servers usually **do** return log-probabilities, which most free hosted
> tiers do not. If a machine with more RAM becomes available, enabling Ollama recovers confidence
> signal S3 with no other change to the project.

### 2.3 Different wire format → one new class

Implement `LLMProvider` and register it. `GoogleGenAIProvider` is the worked example — Gemini uses
`contents` rather than `messages`, calls the assistant turn `model`, and nests config under
`generationConfig`.

```python
# src/pramana/generation/providers/my_llm.py
from pramana.generation.base import (
    Capability, Completion, GenerationRequest, GenerationResponse, Usage,
)

class MyLLMProvider:
    name = "my_llm"

    def __init__(self, *, name="my_llm", default_model="", **kw):
        self.name, self.default_model = name, default_model

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        text = call_my_backend(  # whatever your SDK/transport is
            [(m.role, m.content) for m in request.messages],
            model=request.model or self.default_model,
        )
        return GenerationResponse(                 # normalise into the neutral shape
            completions=[Completion(text=text, finish_reason="stop")],
            model=request.model or self.default_model,
            provider=self.name,
            usage=Usage(),
        )

    def capabilities(self, model): return {Capability.SYSTEM_ROLE}
    def available_models(self):    return [self.default_model]
    def health_check(self):        return True
```

Register the factory:

```python
# src/pramana/generation/registry.py
_ADAPTERS["my_llm"] = _lazy("pramana.generation.providers.my_llm", "MyLLMProvider")
```

Then reference `adapter: my_llm` in YAML. **The pipeline is untouched.**

---

## 3. What the Router Adds

Using `LLMRouter` instead of a bare provider buys four things that matter under free-tier constraints:

| Feature | Why it matters here |
|---|---|
| **Content-addressed cache** | A repeated request never re-spends quota. Re-running an experiment is free; a crashed batch job resumes where it stopped. |
| **Client-side throttling** | Stays under declared limits instead of collecting 429s. Daily counters persist across restarts. |
| **Automatic failover** | When one provider hits its daily cap at 2 a.m., the run continues on the next instead of dying. |
| **Capability negotiation** | An unsupported optional parameter (`logprobs`, `seed`, `n`) is stripped and the capability downgraded, rather than failing the request. |

```python
from pramana.generation import LLMRouter, GenerationRequest, Message

with LLMRouter() as router:                    # reads providers.yaml + .env
    response = router.generate(GenerationRequest(
        messages=(Message("system", "Answer only from the context."),
                  Message("user", "Why was my claim rejected?")),
        model="llama3",                        # alias, resolved per provider
        n=4,                                   # self-consistency samples (S4)
        want_logprobs=True,                    # dropped silently where unsupported
    ))
    print(response.text, response.provider, response.cached)
```

Offline, with no key and no network:

```python
router = LLMRouter(providers=["stub"])
```

---

## 4. Capabilities Are Measured, Not Declared

The `capabilities:` list in YAML is **optimistic**. Free endpoints routinely advertise features they
do not deliver — most importantly `logprobs`, which confidence signal **S3** depends on.

```powershell
python scripts/probe_providers.py
```

This empirically tests each provider for system-role handling, logprobs, multi-sample, seed
determinism, and Hindi/Tamil output quality, then writes `reports/provider_capabilities.json`, which
**overrides** the YAML at runtime.

Two mechanisms keep declarations honest:

1. **Probe-time correction** — `Registry.apply_probe_results()` replaces declared capabilities with
   measured ones and disables unreachable providers.
2. **Runtime downgrade** — if an endpoint rejects a parameter mid-run, the adapter strips it, records
   the downgrade, and retries. One rejection does not lose a twelve-hour job.

`router.guaranteed_capabilities()` returns the **intersection** across live providers — the honest
basis for deciding whether a signal can be relied on for a whole run.

---

## 5. Reproducibility Under Hosted Models

A hosted model can be updated or retired without notice, so **the endpoint is not a stable artefact**.

> **The cache is the reproducible artefact.** Archive `data/cache/generations/` and
> `data/processed/drafts_*.jsonl` alongside the report. Every reported number can then be regenerated
> exactly, even if the provider has changed the model or shut the endpoint down.

Every run manifest records the provider, the exact resolved model string, and the generation date.
Without this, split execution silently destroys reproducibility —
the same config could replay against different drafts.

---

## 6. Verifying an Integration

```powershell
python scripts/check_env.py                          # key seen? registry loads?
python scripts/probe_providers.py --provider NAME    # what does it actually do?
python -m pytest tests/test_generation.py -q         # contracts still hold
```

A new provider is correctly integrated when `check_env.py` lists it as ready, the probe reports it
reachable, and the test suite passes unchanged — the last point being the real check, since it proves
the pipeline did not have to learn anything about the new backend.

---

## 7. Design Rationale (for the viva)

**Why `httpx` directly instead of vendor SDKs?**
One HTTP dependency covers a dozen providers. Eight SDKs would mean eight sets of pinned transitive
dependencies to reconcile — a real problem on a machine with 49 GB of free disk and 7.7 GB of RAM —
and vendor SDKs periodically diverge from the wire format they document.

**Why is `provider` part of the cache key?**
The same nominal model on two hosts can differ in quantisation, serving stack, or silent version. They
must not share a cache entry, or an ablation would compare contaminated results.

**Why throttle client-side when the server already returns 429?**
A 429 costs a round trip plus a backoff, and repeatedly tripping limits can get a key blocked. Staying
just under the line is cheaper and safer.

**Why does a missing logprob become a missing feature rather than a default value?**
Imputing a value would corrupt calibration silently — the confidence model would learn from a constant
that carries no information while appearing to. Refitting without S3 is the honest handling, and the
constraint must be reported with each evaluation.

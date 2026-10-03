# Enterprise pilot runbook

This release supplies controls for a **small, single-tenant, supervised pilot**.
It does not certify accuracy or authorize use of confidential documents with an
external provider. All users of one instance must have the same document access.

## What is implemented

- Bearer authentication on every route except liveness, including OpenAPI.
- Exact trusted Host allowlist (localhost/loopback by default), preventing an
  attacker-controlled Host/Origin pair from bypassing the demo boundary. Set
  PRAMANA_TRUSTED_HOSTS to include the approved TLS proxy hostname before remote
  deployment. Wildcards, schemes and ports are not accepted in this setting.
- Anti-framing and no-referrer headers on responses, including rejected requests.
- Approved UTF-8 `.md`/`.txt` files under language directories; bounded file,
  document, and chunk counts; versioned citations using document hashes.
- An explicit provider allowlist. Pilot mode cannot fall back to the offline stub.
- A single active inference request, a 32 KiB body limit, and bounded correction
  attempts. Busy requests receive 503 with Retry-After.
- At most 32 provider HTTP attempts per request by default, including verification
  and retries. The cooperative 60-second budget caps each next HTTP timeout to
  remaining time. It is **not a hard process deadline**: HTTP timeouts concern
  transport operations, and local encoding/verification can overrun the budget.
- Fail-closed answers: unresolved evidence, unsupported extraction, low confidence,
  or failed repairs lead to a localized abstention. This is conditional on the
  verifier's judgments; verifier mistakes remain possible.
- API errors, malformed verdict labels and empty/truncated completions are not
  converted to factual judgments. The router tries approved backup providers;
  if all fail, inference returns a sanitized 503. Whole-answer coverage and
  question relevance are checked even when the decomposer omits a clause. Short
  answers are checked with their question rather than as an isolated number.
- Contradictions take precedence over support in pilot mode when sources conflict.
- Sensitive exception text is suppressed from API responses and routine logs;
  HTTP client request logging is disabled in pilot mode.
- Generated-text caching and full-corpus browsing are disabled by default for the
  environment-driven pilot configuration. Rate-limit counters remain on disk.
- Readiness reports state the backend, retrieval mode and whether confidence is
  fitted. Readiness does not contact providers or measure their accuracy.

## Prepare the deployment

1. Choose a document owner, pilot users, and an approved model endpoint. Hosted
   endpoints receive query/context/claim text. Use the provider allowlist to prevent
   failover to unapproved destinations. A local OpenAI-compatible endpoint can be
   configured with PRAMANA_PROVIDER_CONFIG.
2. Export approved documents as UTF-8 Markdown/text and arrange them as:

   ```text
   data/pilot/corpus/en/insurance/claims.md
   data/pilot/corpus/hi/insurance/claims.md
   data/pilot/corpus/ta/insurance/claims.md
   ```

   Each configured language must have at least one non-empty document. Set
   PRAMANA_LANGUAGES to a subset for an initial pilot. Do not mix access tiers in
   the same instance. PDF/Word conversion and document approval happen upstream.
3. Copy `.env.pilot.example` to `.env.pilot`. Generate a token with
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`, set
   PRAMANA_API_KEY, choose PRAMANA_PROVIDERS, and add the relevant provider key.
   The two keys have different purposes: the first authenticates clients; the
   second authenticates PRAMANA to its model provider.
4. Run `docker compose up --build -d`. The service binds **127.0.0.1:8000**.
   The image runs as a non-root user with a read-only root filesystem. The compose
   memory budget of 2 GB is for hosted generation and LLM verification.
5. For access beyond the machine, place the service behind the organization's
   TLS reverse proxy or VPN, restrict users, and configure ingress rate and body
   read limits. Do not expose the development server directly to the internet.
   Use **one process/worker and one replica**: concurrency and quota counters are
   not coordinated across workers. Scaling requires shared state and per-user
   authentication, which are outside this pilot implementation.

Docker must be running. Compose configuration can be validated without secrets or
Docker Engine using `docker compose config --no-env-resolution --quiet`.

For a local demo without Docker:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pilot.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
$env:PRAMANA_MODE = 'demo'
$env:PRAMANA_OFFLINE = 'true'
.\.venv\Scripts\python.exe -m uvicorn pramana.api.service:app --host 127.0.0.1 --port 8000
```

For native pilot serving, supply the same environment variables from the example
and set PRAMANA_CORPUS_DIR to the absolute approved corpus directory. Do not use
`--reload` for pilot service.

## Verify access and readiness

The public liveness route is `/v1/health`. All other requests need
`Authorization: Bearer <PRAMANA_API_KEY>`.

`/v1/runtime` exposes safe process counters, configured/resolved model names,
failover/error types and embedding request counts. It does not expose keys or
document text, and is not an account-quota dashboard. Local token reservations
include a prompt estimate and are reconciled with total returned usage rather
than charging the output twice. Failed requests retain conservative reservations.

```powershell
$headers = @{ Authorization = "Bearer $env:PRAMANA_API_KEY" }
Invoke-RestMethod http://127.0.0.1:8000/v1/ready -Headers $headers
$body = @{ query = 'What is the claim submission deadline?'; language = 'en' } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/ask -Method Post -Headers $headers -ContentType 'application/json' -Body $body
```

`/v1/ask` generates and verifies. `/v1/verify` audits an existing answer and
preserves it, including unsupported statements; that endpoint is for operators,
not for automatically publishing answers. Audit responses have no correction
actions. Pilot mode rejects `assurance=false` on `/v1/ask`.

Store `trace_id`, HTTP `x-request-id`, `stop_reason`, action history, and citation
IDs in the organization's approved observability system. These two IDs identify
pipeline work and the HTTP request respectively. Do not log entire requests,
responses, bearer tokens, or document contents by default. This repository does
not provide a persistent audit database or export telemetry to a third party.

## Retrieval and confidence

The recommended low-RAM deployment uses hosted generation and verification.
For hybrid semantic retrieval without local model weights, set
`PRAMANA_API_EMBEDDING_MODEL=gemini-embedding-001` and approve `google` in
`PRAMANA_PROVIDERS`. It uses separate retrieval-document/query task types and
768-dimensional normalized vectors. Query and document text leave the machine;
get document-owner approval first. Embedding quota errors stop the operation,
not silently switch retrieval quality. Startup rebuilds the in-memory index and
may be rate-limited with a large corpus; the model is text-only, not OCR.

The minimum no-embedding deployment uses BM25. Set PRAMANA_DENSE_MODEL to an approved local
multilingual SentenceTransformer model to enable semantic retrieval plus RRF.
Install the NLP extra and CPU PyTorch, mount the model read-only, and raise the
memory allocation. Models should be staged before deployment; optional model
startup can otherwise access the model host. The index shares one CPU encoder
across languages and filters low similarity results; tune the similarity threshold
on representative documents. No real encoder quality measurement is claimed by
the offline tests, which use deterministic vectors.

The optional transformer verifier similarly requires the NLP extra and a staged
model. The base container has neither model weights nor those extra dependencies.

PRAMANA_CONFIDENCE_MODEL can load a fitted confidence JSON artifact. Without one,
confidence is explicitly heuristic. If required features disappear at inference,
the service falls back to an explicitly uncalibrated score instead of silently
imputing values. A language-specific model can only be used with that language.
Self-consistency sampling is disabled in the service; fit the artifact with the
same feature availability as the deployed configuration. Calibrate on development
data and evaluate once on held-out pilot questions.

Use `scripts/fit_confidence.py` to fit attributed human train/dev JSONL rows,
with `qid`, `parallel_id`, `question`, `language`, `split`, `human_reviewed`,
`reviewer`, `fully_supported`, and `features` (the API confidence signal dict).
It rejects train/dev overlap, unknown/nonfinite features and test-set fitting.
Both splits need at least 20 reviews and both label classes; this is a workflow
minimum, not evidence of sufficient statistical power. No human calibration
dataset or approved fitted artifact is supplied with this release.

```powershell
.\.venv\Scripts\python.exe scripts/fit_confidence.py --train data/pilot/train.jsonl --dev data/pilot/dev.jsonl --language en --out data/pilot/confidence-en.json
```

A single language-specific artifact requires a matching single-language pilot.
For multilingual deployment, keep scores explicitly heuristic until a validated
pooled artifact or separate approved per-language instances are available.

## Acceptance gate

Before customer rollout, have domain owners review held-out outputs. Use one JSONL
record per question, with these required fields:

```json
{"qid":"en-heldout-001","language":"en","split":"test","human_reviewed":true,"reviewer":"domain-owner-id","abstained":false,"should_abstain":false,"fully_supported":true,"confidence":0.9,"confidence_calibrated":true,"latency_ms":1800,"baseline_latency_ms":1200}
```

This is a schema example, **not evidence**. `fully_supported` is the human judgment
of the returned answer; `confidence_calibrated` must match the actual deployment.
Include personal-case questions, contradictions, policy exceptions, missing facts,
and retrieval misses. Keep baseline and pilot latency measurements under comparable
cache and load conditions. Do not use stub outputs, synthetic judgments, training
questions or the demo corpus as acceptance evidence.

```powershell
.\.venv\Scripts\python.exe scripts/pilot_acceptance.py --reviews data/pilot/reviews.jsonl --out reports/pilot-acceptance.json
```

Proposed gates, checked separately for each configured language:

| Gate | Proposed threshold |
|---|---|
| Attributed held-out reviews | At least 50 |
| Non-abstained answers | At least 20 |
| Coverage of answerability | Both answerable and unanswerable cases |
| Human-identified unsupported answers | At most 5% of returned answers |
| Over-abstention | At most 20% of answerable cases |
| Abstention recall | At least 90% |
| Confidence | Actually calibrated; ECE at most 0.10 |
| p95 latency relative to baseline | At most 2x |

Agree these thresholds with the pilot owner before collecting the test set. Small
samples do not demonstrate rare-event safety; this gate does not estimate an upper
confidence bound on failure rate. The tool exits nonzero when evidence is missing
or a gate fails. It checks attributed review fields, not the reviewer's identity.

## Operations and rollback

- Start with operator review of all answers. Expand exposure only after acceptance.
- Treat 401 as a credential/configuration issue, 413/422 as invalid input, and 503
  as busy or unavailable. Retry 503 with backoff; do not run an unbounded retry loop.
- Changes to documents require review and service restart, which rebuilds indexes.
  Changed document hashes produce new citation IDs. Archive the approved corpus
  version when retaining answer audit records.
- Re-run acceptance after changing a provider/model, corpus, verifier, thresholds,
  or confidence artifact. Preserve the previous approved image and corpus snapshot.
- Roll back by restoring that image/configuration/corpus snapshot and restarting.
  Disable customer routing first if a serious unsupported answer is observed.
- Rotate the bearer key through environment configuration and restart. This pilot
  has a shared key, not per-user revocation, SSO, role-based document access or
  tenant isolation. Do not use it for users with different document permissions.

## Validation recorded for this change

The current API-only checks and release blockers are recorded in
[VALIDATION.md](VALIDATION.md). The following Docker paragraph
describes an earlier run, not a rebuild of the current source. Docker Engine was
not reachable during the 2026-09-30 hardening run; Compose syntax validation alone
does not verify the container.

The local suite passed 470 tests (two dependency deprecation warnings), and lint passed. The Python wheel and Docker image built successfully. The running demo container was healthy, non-root, and read-only; HTTP verification caught the numeric contradiction and empty-evidence case. Unconfigured pilot startup was rejected. See `reports/pilot-hardening/readiness.json` for the recorded scope. Offline tests exercise
request isolation, access control, byte limits (including chunked bodies), capacity,
budgeted provider calls, failure handling, corpus provenance, retrieval integration,
and pilot acceptance gates. Docker Engine availability, live endpoint availability,
real encoder quality, human correctness and calibration require separate checks.

## Rehearse without enterprise documents or provider credentials

Use the fictional files under `examples/corpus`:

```powershell
docker compose -p pramana-rehearsal -f compose.demo.yaml up --build -d
```

Open http://127.0.0.1:8000. This is local **demo mode** using the offline stub,
not pilot mode or a measurement of model quality. It needs no API key. Stop it
with `docker compose -p pramana-rehearsal -f compose.demo.yaml down` before starting the pilot config
on the same port. The demo and pilot have separate state volumes.

The native live alternative is `.\run_demo.cmd --semantic --port 8765` (Google
key required, optional Groq fallback). Use `--offline` explicitly for fixture
rehearsal. No implicit offline fallback is allowed by this launcher.

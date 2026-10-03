# Validation, release readiness and human review

This page is the single public record of what is tested, what a release must
pass, and what still needs human or deployment evidence. It replaces the former
`12_RELEASE_READINESS.md` and `13_VALIDATION_AND_HUMAN_REVIEW.md`. Fixture checks
and small live smoke runs are never presented as accuracy results.

## 1. Offline engineering baseline

The Windows development run on 2026-10-02 completed 610 offline tests with 81%
line coverage, a clean `ruff` lint and a clean `mypy` type check. CI repeats
the suite on Linux with Python 3.11, 3.12 and 3.13 and on Windows with 3.12,
and fails below 78% coverage. The suite covers the API, retrieval and
correction policies, batched verification, Romanised-query retrieval, provider
failure handling, request limits, security headers, launcher behaviour,
dataset validation, and multilingual fixture cases.

Reproduce from a clean clone:

```text
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m ruff check src scripts tests
python -m mypy src/pramana
python -m pytest -o addopts='' -q -m "not network and not slow" --cov=pramana --cov-fail-under=78
python scripts/check_publication.py
```

The Docker path builds the image without credentials and exercises the health
endpoint and offline HTTP behaviour. CI also scans the image for fixable
critical vulnerabilities (Trivy) and runs CodeQL on the Python sources.

For a controlled local operations check:

```text
python scripts/validate_operations.py --out reports/my-operations-check
```

It uses a test-only deterministic provider; its timings are not live-model
performance and must not be used as an SLA.

## 2. What the fixtures prove

The fixtures contain deliberately planted defects: altered numbers, missing
facts, dropped negation, empty retrieval, and representative Hindi/Tamil
language forms. They show that the implementation reaches the expected policy
branch for known inputs. They do **not** estimate real-world precision, recall,
calibration, multilingual quality, or provider quality.

## 3. Live smoke scope

Optional live checks exercise the configured Google and Groq adapters, provider
validation, quota reservation/reconciliation, and the three UI languages. They
depend on moving model aliases, network availability and quotas, are not CI
gates, and must not be read as a benchmark.

The provider probe on 2026-10-02 (`reports/provider_capabilities.json`) found
both Google (`gemini-3.5-flash-lite`) and Groq (`openai/gpt-oss-20b`) reachable,
honouring the system role and producing English, Hindi and Tamil text; neither
returns token log-probabilities, so confidence signal S3 remains unavailable.
An earlier probe budget of 32 output tokens made the reasoning model look
broken in every language; probe budgets are now large enough for reasoning
models.

## 4. Release gates

Before a tagged release, the commands in §1 must pass, and the Docker image must
build and answer its health endpoint without `.env` files or API keys.
Live-provider checks are opt-in, cost quota, and must record provider model
identifiers and the date.

## 5. Evaluation data and human review

The evaluation set must follow [05_DATASET_SPEC.md](05_DATASET_SPEC.md) and pass
the mechanical checks before any experiment uses it:

```text
python scripts/validate_dataset.py data/eval/questions.jsonl --corpus data/raw
```

Errors (schema, cross-language split leakage, inconsistent parallel groups, gold
evidence absent from the corpus) make a set unusable. Warnings (unreviewed
translations, balance or size below plan) must be reported with any result.
`examples/eval/questions.example.jsonl` is a 13-item worked example against the
bundled fictional corpus; its Hindi and Tamil items are unreviewed drafts.

Collect approved questions and documents before fitting or evaluating a
confidence model. Keep train/dev/test splits disjoint by question and by
parallel family. For each language, record the reviewer, source passage,
answerability, claim-level label, abstention acceptability, and the model
identifiers used. Use at least two annotators per language and report Cohen's
κ (`pramana.evaluation.annotation`).

The tooling requires at least 20 train and 20 dev reviews per language for
fitting and 50 held-out test reviews per language for the acceptance gate.
These are workflow minimums, not adequate power for rare errors.

```text
python scripts/fit_confidence.py --train data/pilot/train-en.jsonl \
  --dev data/pilot/dev-en.jsonl --language en --out data/pilot/confidence-en.json
python scripts/pilot_acceptance.py --reviews data/pilot/test-reviews.jsonl \
  --out reports/human-pilot-acceptance.json
```

The fitting report includes equal-width ECE with a bootstrap 95% interval,
equal-mass (adaptive) ECE, the Brier score and feature weights. Judge the
H3 criterion (ECE ≤ 0.10) against the interval's upper bound, not the point.

Run the comparison systems, including the local mDeBERTa verifier arm, with:

```text
python scripts/run_experiment.py --eval data/eval/questions.jsonl --corpus data/raw \
  --systems vanilla keyword sentence_nli llm_judge pramana pramana_mdeberta
```

## 6. Known gaps before production

- Confidence is heuristic until an independent human-reviewed calibration
  artifact is supplied.
- No universal multilingual accuracy or superiority over a baseline is claimed.
- TLS/SSO/RBAC, tenant/document authorization, ingress limits, threat
  modelling, sustained load, outage/recovery, backup/restore and an independent
  penetration test remain deployment work in the target environment.
- Every provider destination and retention path needs document-owner approval.
- The default service is single-worker, single-tenant, and localhost-only.
- Human review is required for consequential decisions; abstention is not
  proof that an answer is true.
- The arXiv working draft and private review material are excluded from this
  release. Do not quote fixture or smoke results as research accuracy, and do
  not assign invented human labels.

Update this page when a limitation is closed with a reproducible artifact, not
just a code change.

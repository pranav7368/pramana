# Validation and human-review handoff

This document describes the checks that remain after the offline engineering
baseline. It intentionally keeps machine smoke results separate from human
acceptance and production approval.

## Reproducible engineering checks

From a clean clone, run the commands in [VALIDATION.md](VALIDATION.md). The
repository test suite covers API contracts, retrieval/correction policies,
provider failure handling, request limits, security boundaries, launcher
behaviour, and English/Hindi/Tamil fixture cases. The Docker smoke path builds
without credentials and exercises the real health and offline HTTP boundaries.

For a controlled local operations check:

```text
python scripts/validate_operations.py --out reports/my-operations-check
```

This uses a test-only deterministic provider. Its timings are not live-model
performance and must not be used as a customer SLA.

## Human review protocol

Collect new, approved questions and documents before fitting or evaluating a
confidence model. Keep train/dev/test splits disjoint by question and by
paraphrase/parallel family. For each language, record the reviewer, source
passage, answerability, claim-level label, abstention acceptability, and the
configuration/model identifiers used for the answer.

The current tooling expects a minimum workflow of 20 train and 20 dev reviews
per language for fitting, and 50 held-out test reviews per language for the
acceptance gate. These are workflow minimums, not a claim of adequate power for
rare errors. Include supported, contradicted, unverifiable, and abstention
cases. Never relabel development smoke examples as held-out data.

Fit and evaluate only after labels are independently reviewed:

```text
python scripts/fit_confidence.py --train data/pilot/train-en.jsonl \
  --dev data/pilot/dev-en.jsonl --language en \
  --out data/pilot/confidence-en.json
python scripts/pilot_acceptance.py --reviews data/pilot/test-reviews.jsonl \
  --out reports/human-pilot-acceptance.json
```

## Deployment handoff

Before handling real or regulated data, an owner must approve provider
destinations and retention, put remote access behind TLS and an authenticated
proxy, verify tenant/document authorization, and complete load, outage,
backup/restore, monitoring, rollback, and independent security tests in the
target environment. The default service remains single-worker, single-tenant,
and localhost-only.

The software can report evidence and abstain; neither behaviour substitutes for
human judgement in a consequential workflow.

# Release readiness

PRAMANA is ready to share as a reproducible research/pilot codebase. It is not
an unconditional production safety certification and it does not prove that an
LLM will always answer correctly.

## Supported release profile

- API-only generation, claim decomposition, verification, and optional Google
  embeddings; no local model or GPU is required.
- English, Hindi, and Tamil demo paths with fictional documents.
- FastAPI service, browser UI, offline fixtures, Docker Compose profile, and a
  single-worker pilot boundary.
- Bounded provider calls, sanitized errors, request limits, authentication in
  pilot mode, and explicit abstention when evidence is insufficient.

Run the Docker path from the [root README](../README.md) or use the native
[quickstart](../QUICKSTART.md). The [public validation summary](VALIDATION.md)
contains the reproducible commands and the evidence scope.

## Release gates

Before a tagged release, run:

```text
python -m ruff check src scripts tests
python -m pytest -o addopts='' -q -m "not network and not slow"
python scripts/check_publication.py
```

Build the Docker image without `.env` files or API keys and exercise its health
endpoint. Live-provider checks are opt-in, cost quota, and must record the
provider model identifiers and date.

## Not yet a production approval

The following require an owner and deployment-specific evidence:

1. human-reviewed, disjoint train/dev/test data for each supported language;
2. confidence calibration, baseline comparisons, retrieval recall, error
   analysis, and statistically appropriate uncertainty intervals;
3. TLS/SSO/RBAC, tenant/document authorization, ingress limits, and an
   independent threat/penetration review;
4. sustained load, outage/recovery, backup/restore, monitoring, and rollback
   tests in the actual hosting environment;
5. document-owner approval for every provider destination and retention path.

The arXiv working draft and private review material are intentionally excluded
from this repository release. Do not quote fixture or smoke results as research
accuracy, and do not assign invented human labels.

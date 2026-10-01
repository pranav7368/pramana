# Validation and limitations

This page is the public, reproducible summary for the repository. It records
what is tested without turning fixture checks or a small live smoke run into an
accuracy claim.

## Offline baseline

The Windows development run on 2026-10-01 completed 570 tests. The same
offline suite was exercised inside the Linux Docker image. It covers the API,
retrieval and correction policies, provider failure handling, request limits,
security boundaries, launcher behaviour, and multilingual fixture cases.

Reproduce the relevant checks from a clean clone:

```text
python -m pip install -r requirements-pilot.txt
python -m pip install --no-deps -e .
python -m ruff check src scripts tests
python -m pytest -o addopts='' -q -m "not network and not slow"
python scripts/check_publication.py
```

The Docker path additionally builds the image without credentials and exercises
the health endpoint plus offline HTTP behaviour. CI repeats the safe subset on
Linux and Windows.

## What the fixtures prove

The fixtures contain deliberately planted defects: altered numbers, missing
facts, dropped negation, empty retrieval, and representative Hindi/Tamil
language forms. They show that the implementation reaches the expected policy
branch for known inputs. They do **not** estimate real-world precision, recall,
calibration, multilingual quality, or provider quality.

## Live smoke scope

Optional live checks have exercised the configured Google and Groq adapters,
provider validation, quota reservation/reconciliation, and the three supported
UI languages. They depend on moving model aliases, network availability,
provider quotas, and the local environment. Their results are not CI gates and
must not be read as a benchmark.

## Known gaps before production

- Confidence is heuristic until an independent human-reviewed train/dev
  calibration artifact is supplied.
- No universal multilingual accuracy or superiority over a baseline is claimed.
- TLS/SSO/RBAC, multi-tenant isolation, external ingress, threat modelling,
  load/soak testing, and an independent penetration test remain deployment work.
- The default service is single-worker, single-tenant, and localhost-only.
- Human review is required for consequential decisions; abstention is not proof
  that an answer is true.
- The arXiv working draft and private review material are deliberately excluded
  from this software release.

Contributors should update this page when a limitation is closed with a
reproducible artifact, not just a code change.

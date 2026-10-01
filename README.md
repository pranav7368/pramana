# PRAMANA

**API-first multilingual assurance for retrieval-augmented generation (RAG).**

PRAMANA checks generated answers against retrieved evidence, separates supported,
contradicted, and unverifiable claims, and applies conservative correction or
abstention policies. It is designed for reproducible research and a small,
localhost pilot on a low-memory machine: generation, claim decomposition, and
verification use hosted APIs while retrieval and policy decisions run locally.

[![Checks](https://github.com/pranav7368/pramana/actions/workflows/checks.yml/badge.svg)](https://github.com/pranav7368/pramana/actions/workflows/checks.yml)

> **Current status:** the offline pipeline, API, Docker quickstart, security
> boundaries, and pilot controls are implemented and tested. Accuracy,
> calibration, multi-tenant production readiness, and customer acceptance are
> not claimed. See [validation and limitations](docs/VALIDATION.md).

## What is included

- English, Hindi, and Tamil question/answer auditing with language-aware claim handling.
- BM25/vector retrieval, optional Google API embeddings, and evidence citations.
- Provider routing with bounded quotas, timeouts, caching, validation, and failover.
- Conservative actions: accept, regenerate, re-retrieve, prune, or abstain.
- FastAPI service plus a browser demo with upload limits and provenance display.
- Offline fixtures and a 570-test validation baseline; fixture results are not an accuracy benchmark.
- A Docker image that contains the fictional sample corpus and does not require a local model or GPU.

The live profile is intentionally single-worker and single-tenant. It is a
research/pilot foundation, not a turnkey public production service.

## Fastest path: Docker Desktop

Docker Desktop with the Linux engine is the recommended low-RAM setup. The host
only needs Git and Docker; no Python, CUDA, or model download is required.

```powershell
git clone https://github.com/pranav7368/pramana.git
cd pramana
.\docker-run.cmd setup
# Edit .env and add GOOGLE_API_KEY (GROQ_API_KEY is optional failover)
.\docker-run.cmd
```

Open <http://127.0.0.1:8765/>. The first run builds the image and creates the
`pramana-local` Compose project. Later runs reuse the same container with:

```powershell
.\docker-run.cmd start
.\docker-run.cmd status
.\docker-run.cmd logs
.\docker-run.cmd stop
```

Use `.\docker-run.cmd build` after source changes. The helper validates the
Docker engine, waits for `/v1/health`, passes only the selected provider keys,
runs as a non-root user with a read-only root filesystem, and keeps the named
state volume. Uploaded documents are held in memory and must be uploaded again
after a restart. This is localhost-only; it is not a hosted endpoint.

Full details and troubleshooting: [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md).

## Native setup (optional)

Python 3.11–3.13 is supported; Python 3.12 is the CI baseline. The launcher
creates `.venv`, installs the pinned pilot requirements, and never overwrites an
existing `.env`.

```powershell
git clone https://github.com/pranav7368/pramana.git
cd pramana
.\run.cmd setup
# Edit .env and add GOOGLE_API_KEY; GROQ_API_KEY is optional
.\run.cmd
```

Use `.\run.cmd doctor` for read-only checks, `--port 8770` for another port,
`--no-semantic` for sparse retrieval, or `--offline` for the fixture rehearsal.
On macOS/Linux: `sh run.sh setup` and `sh run.sh`. A live provider consumes
provider quota and may incur charges. Never commit keys.

The UI accepts a UTF-8 text/Markdown file or a PDF up to 5 MB and 25 pages.
Scanned PDFs need OCR first. The bundled sample corpus is fictional and exists
for wiring demonstrations only.

## Architecture

```text
question → language/script → retrieval → generation → atomic claims
                                                    ↓
                              evidence/NLI verification → confidence
                                                    ↓
                              accept · repair · prune · abstain
```

The assurance layer is provider-agnostic. A YAML provider entry can point to a
compatible hosted endpoint; provider-specific wire formats use an adapter. See
[LLM integration](docs/10_LLM_INTEGRATION.md) and the
[pilot runbook](docs/11_ENTERPRISE_PILOT.md).

## Reproduce the checks

The default test command is offline and needs no API key:

```powershell
python -m pip install -r requirements-pilot.txt
python -m pip install --no-deps -e .
python -m ruff check src scripts tests
python -m pytest -o addopts='' -q -m "not network and not slow"
python scripts/check_publication.py
```

For the offline Docker smoke path (Linux engine):

```text
docker compose -p pramana-smoke -f compose.demo.yaml up --build -d
python scripts/validate_container_http.py --url http://127.0.0.1:8000 --out reports/container-smoke/report.json
docker compose -p pramana-smoke -f compose.demo.yaml down -v
```

The CI workflow runs lint, offline tests, package installation, publication
checks, and an offline container HTTP smoke test on every push and pull request.
Live-provider validation is deliberately opt-in and is not run in CI.

## Evidence and limitations

The public validation summary records the environment, commands, scope, and
known gaps without presenting fixture smoke tests as model-quality results:
[docs/VALIDATION.md](docs/VALIDATION.md). Important limitations include:

- confidence remains heuristic until an independent human-reviewed calibration artifact is supplied;
- no claim is made about universal multilingual accuracy or benchmark superiority;
- live API quality depends on provider models, quotas, outages, and moving aliases;
- TLS/SSO/RBAC, multi-tenant isolation, external ingress, threat modelling, and production soak testing remain deployment work;
- human review is required before a decision affects a person, customer, or regulated workflow.

The private arXiv working draft is intentionally not part of the software
release. Publish a paper only after its authors, citations, data permissions,
and measured results have been independently reviewed.

## Repository map

| Path | Purpose |
|---|---|
| `src/pramana/` | Library, API, retrieval, verification, correction, and provider adapters |
| `examples/corpus/` | Fictional, redistribution-safe demo documents |
| `tests/` | Offline unit/integration tests and security-boundary checks |
| `scripts/` | Launch validation, release checks, and operational probes |
| `docs/` | Design, deployment, integration, and validation documentation |
| `compose.local.yaml` | Easy local Docker profile |
| `compose.yaml` | Explicit pilot profile; requires its own approved corpus and token |

## Contributing and security

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change.
Security reports and credential disclosures belong in
[SECURITY.md](SECURITY.md), not in a public issue. Do not include API keys,
customer documents, private paper drafts, generated archives, or raw provider
logs in commits.

## License

MIT. See [LICENSE](LICENSE).

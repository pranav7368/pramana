# Changelog

## Unreleased

### Partial answers and interface themes

- A fully supported answer that covers only part of the question is released
  with a fixed notice in the question's language instead of being withheld. The
  whole-answer check now separates faithfulness from completeness: a neutral
  verdict is followed by a FULL / PARTIAL / UNRELATED / UNSUPPORTED judgement,
  and a partial answer is refused when the question depends on the asker's own
  records. Any failure keeps the answer withheld. Responses carry `partial`.
  Behaviour before and after is recorded as P8 in `docs/08_RESULTS.md`.
- The interface has a light, dark and system theme toggle, remembered per
  browser and applied before the first paint.

### Cross-language questions

- An uploaded document's language is detected, and questions in any enabled
  language are answered from it, in the question's language. Retrieval adds a
  translated query variant when the languages differ
  (`PRAMANA_CROSS_LINGUAL_QUERIES`). Verification is told that premise and
  hypothesis may be in different languages.
- One active document per visitor. The interface shows the detected language,
  marks answers drawn from a document in another language, and finds evidence
  passages in any language.
- The Hindi and Tamil sample policies now state the same facts as the English
  one, so the document's language is the only difference between them.
- `scripts/run_crosslingual.py` and `examples/eval/crosslingual.template.jsonl`
  (31 draft items) run every question against every document language. They
  write behaviour tables and a review sheet, and `--score` turns human marks into
  accuracy with bootstrap intervals (`docs/06_EVALUATION_PROTOCOL.md` §5.1).

### Repositories

- The public repository is generated from the private one by
  `scripts/publish_public.py` and `release/public-manifest.txt`. The export
  refuses credentials, private markers, private directories and links to
  unpublished documents. Public links now point to `pranav7368/pramana`, and
  shared engineering docs use neutral reviewer wording.

### Public deployment

- `PRAMANA_MODE=public` and `scripts/serve_demo.py --public` serve the demo to
  anonymous visitors: uploads are private to the uploading browser tab (bounded,
  expiring in-memory sessions), per-visitor and daily usage limits return HTTP 429
  with `Retry-After`, requests queue briefly for the inference slot, and HTTPS
  same-origin writes are accepted behind a TLS proxy. The public hostname comes
  from the platform (`RENDER_EXTERNAL_HOSTNAME`, `PRAMANA_PUBLIC_HOST`).
- Demo uploads no longer replace the shared index for every client; each
  `X-Pramana-Session` gets its own layer over the sample policies.
- `render.yaml` Blueprint for a free Render web service that deploys only after CI
  passes, an opt-in uptime workflow, a CI smoke test of the public container
  command, and `docs/DEPLOYMENT.md`.
- The interface shows a public-demo notice, the remaining checks, links to the API
  docs and source, and explains free-tier cold starts.

### Interface

- Rebuilt the local UI as an evidence desk: source rail, question/audit composer,
  and an evidence ledger. Claims are marked in the answer by verdict; selecting
  one highlights the passage it rests on, the words they share and, for a
  contradiction, the number or negation where the source differs. Light and dark
  themes, keyboard shortcuts, drag-and-drop upload, and a phone layout. Removed
  the classroom-demo wording.

### Assurance quality

- Verification judges a claim against every retrieved chunk in one request
  instead of one request per chunk, keeping per-chunk verdicts and citations.
  Malformed batch replies fall back to per-chunk requests. `LLMNLIBackend(batch=False)`
  restores the old behaviour for ablations; the transformer backend batches too.
- Corrections that lose supported claims are rolled back
  (`CorrectionPolicy.preserve_supported`), so an answer cannot "improve" by
  saying less.
- Romanised Hindi and Tamil questions are rewritten into native script for
  retrieval and fused with the original query (`PRAMANA_TRANSLITERATE_QUERIES`).
- Optional multilingual cross-encoder reranking (`PRAMANA_RERANKER_MODEL`),
  with scores kept on the [0, 1] scale the policy thresholds expect.

### Evaluation

- `pramana_mdeberta` experiment arm isolates the local XNLI verifier.
- Calibration reports add equal-mass ECE, the Brier score, bootstrap 95%
  intervals and feature weights.
- `scripts/validate_dataset.py` checks the question set against the dataset
  specification, including split leakage across languages, plus a worked example
  in `examples/eval/`.

### Operations and security

- Content-Security-Policy on every response: the demo pins its inline script by
  hash, data responses deny everything.
- `/v1/metrics` (Prometheus text), `PRAMANA_LOG_FORMAT=json` with request ids,
  and reranker/transliteration status in `/v1/ready`.
- CI: mypy, a 78% coverage gate, Python 3.11–3.13, CodeQL and a Trivy image
  scan. Version is single-sourced from `pramana.__version__`.

### Fixes

- A provider registry without `router.failover_on` crashed on load (a slots
  dataclass exposes a descriptor, not the default, on the class).
- The provider probe used a 32-token budget that reasoning models spend before
  answering, falsely reporting no system-role or multilingual support; it also
  crashed when writing outside the repository.
- Removed dead ranking code from the verifier and stale `type: ignore` comments.

### Earlier in this cycle

- Added a reproducible Docker-first quickstart for the API-only local profile.
- Added publication hygiene, security, contribution, and validation documents.
- Added CI checks for linting, offline tests, package installation, and the
  no-credential container smoke path.
- Kept research drafts, generated exports, raw provider logs, and credentials
  outside the software release.

## 0.1.0

- Initial research and pilot implementation of multilingual RAG assurance,
  evidence verification, correction policies, and provider adapters.

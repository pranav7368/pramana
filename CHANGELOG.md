# Changelog

## Unreleased

### Fixes

- Answers that mix in another language's script (Devanagari inside a Tamil
  answer, for example) are regenerated once with the stray script named, then
  rewritten into the answer's own script without the source context; an answer
  that is still mixed is withheld with the stop reason `language_mismatch`.
  The claim checker compares meaning and could not see this fault.
- The personal-case check ("does this question need the asker's own
  records?") runs for every released answer, not only partial ones
  (`personal_case`). A prompt change had made "how much fine do I owe?" be
  answered with the general rule.
- An entailed answer is now also checked for coverage, so an answer to part of
  a question always carries the partial notice.
- Pruning rebuilds the answer from its supported claims when a flagged claim has
  no text span. The live decomposer rewrites claims, so pruning used to return
  the answer unchanged and the answer was withheld. A connective left at the end
  of a sentence ("..., and") is removed.
- The generation prompt names the answer language, asks for translation rather
  than copying, keeps qualifiers such as "more than", and asks to answer only
  the covered parts of a multi-part question.
- Provider layer: a provider's "quota spent" signal now blocks it for 30
  minutes instead of until the next UTC midnight (Google resets at midnight
  Pacific time); the limiter's state file is written atomically; and a 429 is
  read as a spent daily quota only when it names a daily period, because Google
  words its per-minute throttle as "quota" too.
- The correction latency budget now counts only time spent correcting. A
  verification on free tiers takes 20-40 s, so a clock started before it had
  always run out when a correction was needed, and correctable answers were
  withheld.
- A word that mixes Latin letters into Hindi or Tamil script ("पॉolicy") is
  treated like a stray script: regenerate, rewrite, or withhold.
- Google's free tier allows 500 requests a day for gemini-3.5-flash-lite
  (measured from its 429 body); the configuration said 1500. The limiter now
  waits until the reset time a provider names ("retry in 15h36m").
- The claim decomposer no longer turns a description inside a noun phrase
  ("rejected claims") into a separate claim ("the claim was rejected"). Such a
  claim is unverifiable against a policy, survived every prune because it was
  extracted again, and got correct answers withheld.
- The verification judge is told that a rule restated in an equivalent form is
  entailment, not contradiction ("must be submitted within 30 days" against
  "rejected if submitted more than 30 days after"). Checked live: equivalent
  rule entailed; wrong number, inverted wrong number and flipped negation still
  contradicted; unrelated claim neutral.
- The decomposer keeps a number or limit in the same claim as the thing it
  applies to. "Covers dental treatment up to 25,000 a year" had been split into
  "covers dental treatment" and "provides up to 25,000 a year"; the second was
  unverifiable on its own, pruning removed it, and the answer lost its amount.
- The personal-case judge has examples in English, Hindi and Tamil, and a
  Hindi or Tamil question that it judges general is asked again in a literal
  English translation that keeps every reference to the asker; either "yes"
  counts. On 22 October the judge had answered "how much fine do I owe for my
  overdue book?" with the general rule in Hindi and Tamil, not in English. A
  live check of eleven questions (three personal, eight general) then gave the
  expected label for all of them.
- The personal-case judge now applies a concrete test: could the exact answer
  be given to anyone who asks, without a detail of the asker's own case? A
  definition with examples had not stopped Gemini from answering "how much fine
  do I have to pay for the book I returned late?" with the general rule.
- A personal question is answered only when the document is a record of the
  asker's own case (a letter or statement about them). This used to be decided
  by the coverage judgement, which calls a general rule a full answer.
- The personal-case judge is told that a named plan, policy or product code is
  not the asker's own record. Gemini had withheld "how much does policy PX-4471
  cover for dental treatment?" in Hindi and Tamil as a personal question.
- `scripts/run_crosslingual.py --retry-errors SECONDS` retries a pair once
  after a provider error, so a free-tier fault is not recorded as behaviour.
- The Hindi and Tamil sample policies said "30 days after discharge" where the
  English one says "more than 30 days after"; they now match, and the example
  gold chunk ids were updated.
- Public mode accepted uploads only up to the 32 KB JSON body limit, so most
  real PDFs were refused with 413 on the hosted demo. Uploads now use the 5 MB
  upload cap in both demo and public mode, with a regression test.

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

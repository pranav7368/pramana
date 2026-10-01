# Contributing to PRAMANA

Thank you for helping improve the project. Small, reviewable pull requests are
preferred.

## Before opening a pull request

1. Explain the behaviour or defect and the user-visible impact.
2. Add or update an offline test for a behavioural change.
3. Run the lint and test commands from the README.
4. Run `python scripts/check_publication.py`.
5. Do not include API keys, customer data, private paper drafts, generated
   archives, raw provider logs, or model caches.

The default test suite must remain runnable without network access, provider
credentials, a GPU, or a local model. Live-provider tests are opt-in and must
be explicitly marked `network`.

## Design expectations

- Preserve the distinction between `SUPPORTED`, `CONTRADICTED`, and
  `UNVERIFIABLE`.
- Prefer fail-closed behaviour when evidence, quotas, or provider responses are
  invalid.
- Keep provider keys and user documents out of logs and error responses.
- Document a limitation when a change cannot be validated offline.

Maintainers may request a threat-model note or a human-review plan for changes
that affect correction, authentication, document access, or confidence scores.

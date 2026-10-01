# Security policy

## Scope

PRAMANA is a research and pilot codebase. The supported security boundary is a
localhost deployment with one worker and one tenant. The project does not claim
to provide production TLS termination, SSO, RBAC, multi-tenant isolation, or a
completed external penetration test.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability or exposed
credential. Contact the repository owner privately through GitHub's security
reporting channel (or the private contact associated with the repository) and
include:

- affected commit or version;
- a minimal reproduction with secrets and personal data removed;
- impact and a suggested mitigation, if known.

If an API key was ever committed or copied into a log, revoke and rotate it
immediately before investigating the repository history. The publication check
in `scripts/check_publication.py` is a guardrail, not a replacement for secret
scanning or incident response.

## Safe operation

Keep `.env` files, approved corpora, uploaded documents, provider responses, and
raw evaluation logs outside commits. Put an authenticated reverse proxy and
TLS in front of any non-local deployment, and complete an independent security
review before handling real user or regulated data.

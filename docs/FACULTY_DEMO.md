# Faculty demonstration guide

## Start

From the repository root:

```powershell
.\run.cmd
```

Open http://127.0.0.1:8765/. The command uses `.env` automatically: Google first,
then Groq. Without a key it stops with a setup message.
Google API embeddings enable automatically with a Google key; use `--no-semantic`
for sparse-only search. [First-time setup](../QUICKSTART.md) needs just one key.
Use `.\run.cmd --offline` to rehearse without model network calls. The first run may
create `.venv` and install dependencies. Keep this terminal open during the demo;
Ctrl+C stops the server. The service binds `127.0.0.1` and uses one worker.

## Seven-minute panel sequence

Use a fictional, text-based PDF that contains a few checkable facts, such as:

Bring an approved fictional, text-based PDF containing a few checkable facts.
Generated PDFs and private review packets are intentionally excluded from the
software release; the bundled corpus can also be restored from the UI for a
wiring-only rehearsal.

- Claims submitted more than **30 days** after discharge are rejected.
- Rejected claims may be appealed within **60 days** of the rejection notice.
- Maternity benefits are unavailable in the first policy year.
- Policy PX-4471 covers dental treatment up to **Rs. 25,000** per year.

Do not use personal or confidential records with a hosted model provider. A
fictional PDF is easier for a panel to inspect and compare against the answer.

1. **Upload (minute 0–1).** Choose English, upload the PDF, open **Preview indexed
   text**, and point to its page and passage count. The demo converts text to
   page-labelled chunks and builds a
   fresh English search index in memory. Hindi and Tamil sample indexes remain.
2. **Grounded answer (minute 1–2).** Ask “How long do I have to appeal a rejected
   claim?” Open the retrieved passage showing 60 days. Explain that the final
   answer's claims are checked against that evidence.
3. **Planted contradiction (minute 2–4).** Select **Audit a draft**. Use the same
   question and enter “Rejected claims may be appealed within 90 days of the
   rejection notice.” Show `CONTRADICTED` and the 60-day source passage. State
   plainly that this is a deliberately written test answer. Audit preserves it;
   it does not correct it.
4. **Correction pipeline (minute 4–5).** Click **Ask PRAMANA this question** to
   run generation, verification and policy actions. Explain only the actions
   actually shown in the trail; a live model may give the right answer first.
5. **Missing information (minute 5–6).** Ask for a WiFi password not present in
   the PDF. Show abstention if the system withholds an answer. Avoid claiming that
   one example proves all unanswerable questions are handled.
6. **Multilingual path (minute 6–7).** Restore the sample corpus, switch to Hindi
   or Tamil, and use one of the prepared example questions. The current retriever
   uses separate indexes per language; English-only uploads do not provide a
   guaranteed cross-language document search.

If the live provider is rate-limited, use `--offline` and the built-in sample
questions to show the interface and deterministic audit. Offline generation uses
fixtures, so never describe those answers as live model results. The planted
wrong answer can also be audited against an uploaded PDF in offline mode.

## What the interface means

- **SUPPORTED**: the verifier judges that the retrieved text entails a claim.
- **CONTRADICTED**: the verifier judges that the text refutes a claim.
- **UNVERIFIABLE**: the retrieved text does not establish the claim.
- **Audit a draft**: checks submitted text without changing it.
- **Ask a question**: runs retrieval, generation, claim checking, confidence
  scoring, and a bounded correction policy.
- **Confidence**: heuristic unless a fitted calibration artifact is configured.
  It is not a guaranteed probability that the answer is correct.

The green cited passages are source text associated with a final claim. The
viewer should still inspect the passage: an attached citation can be wrong.
Demo output is a demonstration, not a research accuracy or pilot acceptance
measurement. The latter requires held-out human-reviewed cases and baselines.

## Upload behavior and limitations

The faculty endpoint supports text PDFs (5 MB, 25 pages, 120,000 extracted
characters), UTF-8 `.txt`, and `.md`. It retains page number, filename and SHA-256
in chunk metadata. The upload is held in process memory and replaces the active
index for the selected language; it disappears on restart or **Restore sample**
for that language.
Image-only/scanned PDFs need OCR before upload. In live mode the question and
retrieved text are sent to the selected model provider. The upload route and UI
are disabled in pilot mode; pilot documents still require upstream approval and
static ingestion according to [the pilot runbook](11_ENTERPRISE_PILOT.md).

Before presenting, run all four prepared questions once against the actual PDF
and chosen provider. Keep the PDF visible on another tab so the panel can compare
source text. If the provider changes its answer, describe the observed output,
not a prewritten success claim.

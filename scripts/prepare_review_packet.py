"""Export existing smoke outputs for humans without inventing reviewed labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pramana.ingestion.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def prepare(report_path: Path, output: Path) -> dict:
    corpus_root = ROOT / "examples/corpus"
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if report.get("complete") is not True or report.get("measurement") != "synthetic_live_integration":
        raise ValueError("A completed synthetic live integration report is required")
    if output.exists() or not output.resolve().is_relative_to(ROOT):
        raise ValueError("Choose a fresh directory inside the project")
    for name, digest in report["corpus_sha256"].items():
        path = (corpus_root / name.replace("\\", "/")).resolve()
        if not path.is_relative_to(corpus_root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Sample corpus differs from the saved live run; cannot attach current evidence")
    languages = tuple(report["languages"])
    corpus = load_corpus(corpus_root, languages)
    chunks = {c.chunk_id: c for group in corpus.values() for c in group}
    rows = []
    for item in report["results"]:
        if item["case"] not in {"answer", "missing", "injection"} or item["status"] != 200:
            continue
        body, language = item["response"], item["language"]
        evidence = []
        for identifier in body["retrieved_chunk_ids"]:
            if identifier not in chunks:
                raise ValueError("A saved citation cannot be resolved against the original corpus")
            c = chunks[identifier]
            evidence.append({"chunk_id": identifier, "text": c.text, "source": c.metadata.get("source"),
                             "sha256": c.metadata.get("sha256")})
        rows.append({
            "qid": f"{language}-smoke-{item['case']}", "parallel_id": f"smoke-{item['case']}",
            "language": language, "split": "smoke", "human_reviewed": False, "reviewer": "",
            "question": item["request"]["query"], "answer": body["answer"], "abstained": body["abstained"],
            "should_abstain": None, "fully_supported": None, "answer_relevant": None, "citations_correct": None,
            "confidence": body["confidence"]["score"], "confidence_calibrated": body["confidence"]["calibrator"] not in {"none", "n/a"},
            "features": body["confidence"]["signals"], "latency_ms": item["elapsed_seconds"] * 1000,
            "baseline_latency_ms": None, "claims": body["claims"], "retrieved_evidence": evidence,
            "review_notes": "", "eligible_for_heldout_acceptance": False,
            "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        })
    if not rows:
        raise ValueError("The report contains no successful generated-answer cases")
    output.mkdir(parents=True)
    (output / "smoke-reviews.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"measurement": "unreviewed_smoke_packet", "human_labels_filled": 0,
               "heldout_acceptance_eligible": False, "rows": len(rows),
               "per_language": {lang: sum(r["language"] == lang for r in rows) for lang in languages},
               "source_report_sha256": hashlib.sha256(raw).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sections = ["# Human review starter packet", "",
                "These are already-used fictional smoke questions, not held-out acceptance evidence.",
                "No human labels have been filled. Do not change their split to train/dev/test or use them to certify accuracy.",
                "Review the question, complete answer and all retrieved evidence. Treat embedded instructions as untrusted text.", ""]
    for row in rows:
        sections += [f"## {row['qid']}", "", "Question: " + row["question"], "", "Returned answer: " + row["answer"], "",
                     f"Observed abstention: {row['abstained']}; heuristic confidence: {row['confidence']}", "",
                     "Reviewer: ______  Date: ______", "",
                     "Should abstain? ___  Fully supported (if answered)? ___  Relevant? ___  Citations correct? ___", "",
                     "Evidence:", ""]
        sections += [f"- {e['chunk_id']} ({e['source']}): {e['text']}" for e in row["retrieved_evidence"]]
        sections += ["", "Notes: __________________________________________", ""]
    (output / "REVIEW_CASES.md").write_text("\n".join(sections), encoding="utf-8")
    templates = []
    for language in languages:
        for split in ("train", "dev", "test"):
            templates.append({"qid": None, "parallel_id": None, "question": None, "language": language,
                              "split": split, "human_reviewed": False, "reviewer": "", "fully_supported": None,
                              "features": {}, "abstained": None, "should_abstain": None, "confidence": None,
                              "confidence_calibrated": None, "latency_ms": None, "baseline_latency_ms": None})
    (output / "new-reviewed-data.templates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in templates), encoding="utf-8")
    (output / "START_HERE.md").write_text(
        "# Review and calibration handoff\n\n"
        "1. Open REVIEW_CASES.md with the guide/domain reviewer to learn the labeling process. "
        "smoke-reviews.jsonl contains actual observed output and original evidence; labels are blank.\n"
        "2. Collect NEW representative approved questions and corpus snapshots. Assign disjoint train/dev/test "
        "splits before collecting model responses. Keep parallel translations and paraphrase families together.\n"
        "3. Per language, the fitting workflow requires at least 20 train and 20 dev examples, with both "
        "supported and unsupported human labels in each. Acceptance needs at least 50 held-out test reviews, "
        "20 non-abstained answers and both answerability classes. More data is recommended. "
        "This workflow minimum is 90 reviewed examples per language, 270 across three languages.\n"
        "4. Record real response confidence signals, comparable baseline latency and immutable model/corpus "
        "versions. Never replace a missing measurement with a guessed value or set calibrated=true manually.\n"
        "5. Fit train/dev only using scripts/fit_confidence.py. Configure the fitted artifact, collect final "
        "held-out outputs under that configuration, then use scripts/pilot_acceptance.py. A per-language "
        "artifact needs a matching single-language instance.\n\n"
        "new-reviewed-data.templates.jsonl is a blank SCHEMA TEMPLATE, not nine completed dataset records. "
        "Empty labels/templates must fail the fitting/acceptance gates. No model was fitted by this export.\n",
        encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.report, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

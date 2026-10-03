#!/usr/bin/env python
"""Cross-language experiment: each question answered from the document in every language.

An exploratory extension of RQ4 (``docs/06_EVALUATION_PROTOCOL.md`` §5.1). For
every question and every document language, the deployed pipeline answers with
the question routed to that document. The sample policies state the same facts
in English, Hindi and Tamil, so a cross pair (Hindi question, English document)
differs from its native pair (Hindi question, Hindi document) only in the
document's language.

Without human labels this measures *behaviour* against the dataset's draft
expectations: answering answerable questions, abstaining on unanswerable ones,
releasing only supported claims. Whether an answer is *correct* goes to
``review.csv`` for a person to mark, and ``--score`` turns those marks into
accuracy with bootstrap intervals. Nothing here is a benchmark result until the
items and the marks have been reviewed by a human.

    python scripts/run_crosslingual.py                      # live, all pairs
    python scripts/run_crosslingual.py --offline --limit 2  # wiring check, no keys
    python scripts/run_crosslingual.py --score reports/crosslingual-<run>/review.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pramana.evaluation.dataset import load_questions  # noqa: E402
from pramana.evaluation.metrics import bootstrap_ci  # noqa: E402
from pramana.generation.budget import request_budget  # noqa: E402
from pramana.retrieval.hybrid import MultilingualRetriever  # noqa: E402
from pramana.schemas import LANGUAGES  # noqa: E402

DEFAULT_EVAL = ROOT / "examples" / "eval" / "crosslingual.template.jsonl"
DEFAULT_CORPUS = ROOT / "examples" / "corpus"
MARKS = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
REVIEW_FIELDS = [
    "review_id", "qid", "condition", "question_language", "document_language", "expected",
    "question", "reference_answer", "system_answer", "stop_reason", "claim_verdicts",
    "evidence_excerpt", "human_mark", "reviewer", "notes",
]


def expected_behaviour(row: dict[str, Any], document_language: str) -> str:
    """What a correct system does for this question against this document."""
    evidence = row.get("evidence_languages")
    answerable = (row["answerability"] in {"A", "B"} if evidence is None
                  else document_language in evidence)
    if not answerable:
        return "abstain"
    return "partial" if row["answerability"] == "B" else "answer"


def select_rows(rows: list[dict[str, Any]], languages: list[str], limit: int | None) -> list[dict[str, Any]]:
    """Keep whole parallel groups so every limited run still has native and cross pairs."""
    chosen = [r for r in rows if r["language"] in languages]
    if limit is None:
        return chosen
    keep = list(dict.fromkeys(r["parallel_id"] for r in chosen))[:limit]
    return [r for r in chosen if r["parallel_id"] in keep]


def reference_answers(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Each row's gold answer, or its English parallel's, marked as such."""
    english = {r["parallel_id"]: r.get("gold_answer", "") for r in rows if r["language"] == "en"}
    refs = {}
    for r in rows:
        own = r.get("gold_answer", "")
        fallback = english.get(r["parallel_id"], "")
        refs[r["qid"]] = own or (f"(en reference) {fallback}" if fallback else "(should abstain)")
    return refs


def build(offline: bool, corpus: Path):
    from dotenv import load_dotenv

    from pramana.api.runtime import build_pipeline
    from pramana.config.settings import Settings

    load_dotenv(ROOT / ".env", override=False)
    keys = {name: bool(os.getenv(f"{name.upper()}_API_KEY", "").strip()) for name in ("google", "groq")}
    providers = tuple(name for name, present in keys.items() if present)
    if not offline and not providers:
        raise SystemExit("Set GOOGLE_API_KEY or GROQ_API_KEY in .env, or pass --offline for a wiring check.")
    settings = Settings(
        mode="demo", corpus_dir=corpus, offline=offline, providers=() if offline else providers,
        api_embedding_model="gemini-embedding-001" if keys["google"] and not offline else "",
        cache_enabled=not offline, provider_retries=2, provider_timeout_s=60,
        max_provider_calls=48, request_budget_s=180,
    )
    return build_pipeline(settings=settings)


def run_pair(pipeline, row: dict[str, Any], document_language: str, chunk_text: dict[str, str],
             reference: str) -> dict[str, Any]:
    base = pipeline.retriever
    routed = MultilingualRetriever(
        retrievers=dict(base.retrievers), fallback=base.fallback, query_variants=base.query_variants,
        route_to=document_language, query_translator=base.query_translator,
    )
    record: dict[str, Any] = {
        "qid": row["qid"], "parallel_id": row["parallel_id"], "question_language": row["language"],
        "document_language": document_language, "script": row.get("script", "native"),
        "condition": "native" if row["language"] == document_language else "cross",
        "expected": expected_behaviour(row, document_language),
        "answerability": row["answerability"], "question": row["question"], "reference_answer": reference,
    }
    try:
        with request_budget(48, 180):
            result = replace(pipeline, retriever=routed).run(row["question"], language=row["language"])
    except Exception as exc:
        return {**record, "error": type(exc).__name__, "answer": "", "abstained": False, "stop_reason": "error",
                "n_claims": 0, "supported": 0, "contradicted": 0, "unverifiable": 0, "verdicts": [],
                "retrieved_chunk_ids": [], "evidence_excerpt": "", "latency_ms": 0.0}
    detection = result.detection
    verdicts = [v.verdict.value for v in detection.claim_verdicts]
    top = result.retrieved_chunk_ids[:1]
    return {
        **record, "error": None, "answer": result.final_answer, "abstained": result.abstained,
        "stop_reason": result.stop_reason, "n_claims": detection.n_claims,
        "supported": verdicts.count("SUPPORTED"), "contradicted": verdicts.count("CONTRADICTED"),
        "unverifiable": verdicts.count("UNVERIFIABLE"), "verdicts": verdicts,
        "retrieved_chunk_ids": result.retrieved_chunk_ids,
        "evidence_excerpt": chunk_text.get(top[0], "")[:300] if top else "",
        "latency_ms": round(result.total_latency_ms, 1),
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"count": numerator, "of": denominator, "rate": round(numerator / denominator, 4) if denominator else None}


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Behaviour against the draft expectations, by condition and by language pair."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[f"condition:{r['condition']}"].append(r)
        groups[f"pair:{r['question_language']}<-{r['document_language']}"].append(r)
    summary: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        ok = [r for r in rows if not r["error"]]
        answer = [r for r in ok if r["expected"] in {"answer", "partial"}]
        abstain = [r for r in ok if r["expected"] == "abstain"]
        released = [r for r in ok if not r["abstained"] and r["n_claims"]]
        summary[name] = {
            "runs": len(rows),
            "errors": len(rows) - len(ok),
            "answered_when_answerable": _rate(sum(not r["abstained"] for r in answer), len(answer)),
            "abstained_when_unanswerable": _rate(sum(r["abstained"] for r in abstain), len(abstain)),
            "released_with_all_claims_supported": _rate(
                sum(r["supported"] == r["n_claims"] for r in released), len(released)),
            "released_with_a_contradicted_claim": _rate(sum(r["contradicted"] > 0 for r in released), len(released)),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in ok) / len(ok), 1) if ok else None,
        }
    return summary


def write_review(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for n, r in enumerate(records, 1):
            writer.writerow({
                "review_id": n, "qid": r["qid"], "condition": r["condition"],
                "question_language": r["question_language"], "document_language": r["document_language"],
                "expected": r["expected"], "question": r["question"], "reference_answer": r["reference_answer"],
                "system_answer": r["answer"] or f"(error: {r['error']})", "stop_reason": r["stop_reason"],
                "claim_verdicts": " ".join(r["verdicts"]), "evidence_excerpt": r["evidence_excerpt"],
                "human_mark": "", "reviewer": "", "notes": "",
            })


def score(review: Path) -> dict[str, Any]:
    """Accuracy from human marks: correct = 1, partial = 0.5, incorrect = 0."""
    with review.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unmarked = [r for r in rows if not r["human_mark"].strip()]
    invalid = [r["review_id"] for r in rows if r["human_mark"].strip() and r["human_mark"].strip().lower() not in MARKS]
    if invalid:
        raise SystemExit(f"human_mark must be one of {sorted(MARKS)}; check review_id {', '.join(invalid[:10])}")
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        mark = r["human_mark"].strip().lower()
        if not mark:
            continue
        value = MARKS[mark]
        groups[f"condition:{r['condition']}"].append(value)
        groups[f"pair:{r['question_language']}<-{r['document_language']}"].append(value)
    result: dict[str, Any] = {"marked": len(rows) - len(unmarked), "unmarked": len(unmarked), "groups": {}}
    for name, values in sorted(groups.items()):
        ci = bootstrap_ci(values)
        result["groups"][name] = {"n": len(values), "accuracy": round(ci.point, 4),
                                  "ci95": [round(ci.lower, 4), round(ci.upper, 4)]}
    return result


def manifest(args, rows: list[dict[str, Any]], router, records: list[dict[str, Any]], offline: bool) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "exploratory cross-language extension of RQ4; behaviour against draft labels, not accuracy",
        "git_commit": commit,
        "dataset": str(args.eval), "dataset_sha256": hashlib.sha256(args.eval.read_bytes()).hexdigest(),
        "questions": len(rows), "runs": len(records),
        "questions_human_reviewed": sum(bool((r.get("qc") or {}).get("human_reviewed")) for r in rows),
        "document_languages": args.doc_languages, "offline": offline,
        "providers": [{"name": b.spec.name, "model": b.spec.default_model} for b in getattr(router, "_bound", [])],
        "router_counters": asdict(router.stats) if getattr(router, "stats", None) else {},
    }


def _cell(metric: dict[str, Any]) -> str:
    return f"{metric['count']}/{metric['of']}" if metric["of"] else "-"


def print_summary(summary: dict[str, Any]) -> None:
    print("\nBehaviour against draft expectations (not accuracy):")
    print(f"  {'group':<22}{'runs':>5}{'answered':>12}{'abstained':>12}{'all supp.':>12}{'errors':>8}")
    for name, s in summary.items():
        print(f"  {name:<22}{s['runs']:>5}{_cell(s['answered_when_answerable']):>12}"
              f"{_cell(s['abstained_when_unanswerable']):>12}"
              f"{_cell(s['released_with_all_claims_supported']):>12}{s['errors']:>8}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="questions JSONL with parallel_id groups")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus directory with en/hi/ta subfolders")
    ap.add_argument("--languages", nargs="+", default=list(LANGUAGES), help="question languages to run")
    ap.add_argument("--doc-languages", nargs="+", default=list(LANGUAGES), help="document languages to route to")
    ap.add_argument("--limit", type=int, help="first N parallel question groups only")
    ap.add_argument("--offline", action="store_true", help="stub provider; checks wiring, measures nothing")
    ap.add_argument("--out", type=Path, help="output directory (default reports/crosslingual-<time>)")
    ap.add_argument("--score", type=Path, help="score a reviewed review.csv instead of running")
    args = ap.parse_args(argv)

    if args.score:
        result = score(args.score)
        out = args.score.with_name("score.json")
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nWrote {out}. Accuracy covers marked rows only ({result['unmarked']} unmarked).")
        return 0

    rows = select_rows(load_questions(args.eval), args.languages, args.limit)
    if not rows:
        raise SystemExit("No questions selected.")
    pipeline, router, offline = build(args.offline, args.corpus)
    chunk_text = {c.chunk_id: c.text for r in pipeline.retriever.retrievers.values() for c in r.chunks}
    refs = reference_answers(load_questions(args.eval))
    out = args.out or ROOT / "reports" / f"crosslingual-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = len(rows) * len(args.doc_languages)
    try:
        with (out / "raw.jsonl").open("w", encoding="utf-8") as raw:
            for row in rows:
                for document_language in args.doc_languages:
                    record = run_pair(pipeline, row, document_language, chunk_text, refs[row["qid"]])
                    records.append(record)
                    raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raw.flush()
                    status = record["error"] or record["stop_reason"]
                    print(f"  [{len(records)}/{total}] {row['qid']} <- {document_language}: {status}", flush=True)
    finally:
        router.close()
    summary = summarise(records)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(manifest(args, rows, router, records, offline), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    write_review(records, out / "review.csv")
    print_summary(summary)
    print(f"\nWrote {out}. Mark human_mark in review.csv (correct / partial / incorrect), "
          "then run --score on it.")
    if offline:
        print("Offline run: stub answers check the wiring only and measure nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

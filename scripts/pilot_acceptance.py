"""Evaluate human-reviewed, held-out pilot answers. Never certifies from smoke data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from pramana.confidence.fusion import expected_calibration_error
from pramana.evaluation.metrics import abstention_metrics


def assess(rows: list[dict], languages=("en", "hi", "ta"), minimum=50) -> dict:
    seen = set()
    for row in rows:
        key = (row["qid"], row["language"])
        if key in seen:
            raise ValueError("Duplicate question/language review")
        seen.add(key)
        if row["split"] != "test" or row["human_reviewed"] is not True or not row["reviewer"].strip():
            raise ValueError("Only attributed human reviews of held-out test answers are accepted")
        if row["language"] not in languages:
            raise ValueError("Review language is outside the pilot scope")
        for name in ("abstained", "should_abstain", "fully_supported", "confidence_calibrated"):
            if type(row[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("confidence", "latency_ms", "baseline_latency_ms"):
            if not isinstance(row[name], (int, float)) or not math.isfinite(row[name]):
                raise ValueError(f"Invalid {name}")
        if not 0 <= row["confidence"] <= 1 or row["latency_ms"] < 0 or row["baseline_latency_ms"] <= 0:
            raise ValueError("Invalid confidence or latency range")
    per_language = {}
    for language in languages:
        group = [r for r in rows if r["language"] == language]
        answered = [r for r in group if not r["abstained"]]
        abst = abstention_metrics([r["abstained"] for r in group], [r["should_abstain"] for r in group])
        unsupported = sum(not r["fully_supported"] for r in answered) / len(answered) if answered else None
        ece = expected_calibration_error([r["confidence"] for r in answered], [int(r["fully_supported"]) for r in answered]) if answered else None
        ratios = sorted(r["latency_ms"] / r["baseline_latency_ms"] for r in group)
        p95_ratio = ratios[max(0, math.ceil(len(ratios) * 0.95) - 1)] if ratios else None
        gates = {
            "reviewed_questions_at_least_50": len(group) >= minimum,
            "answered_questions_at_least_20": len(answered) >= 20,
            "answerable_and_unanswerable_present": any(r["should_abstain"] for r in group) and any(not r["should_abstain"] for r in group),
            "unsupported_answers_at_most_5pct": unsupported is not None and unsupported <= 0.05,
            "over_abstention_at_most_20pct": bool(group) and abst.over_abstention_rate <= 0.20,
            "abstention_recall_at_least_90pct": bool(group) and abst.recall >= 0.90,
            "calibrated_confidence": bool(answered) and all(r["confidence_calibrated"] for r in answered),
            "ece_at_most_10pct": ece is not None and ece <= 0.10,
            "p95_latency_ratio_at_most_2": p95_ratio is not None and p95_ratio <= 2.0,
        }
        per_language[language] = {
            "reviewed": len(group), "answered": len(answered),
            "unsupported_answer_rate": unsupported, "ece": ece,
            "over_abstention_rate": abst.over_abstention_rate,
            "abstention_recall": abst.recall, "p95_latency_ratio": p95_ratio,
            "gates": gates, "passed": all(gates.values()),
        }
    return {
        "passed": bool(per_language) and all(v["passed"] for v in per_language.values()),
        "per_language": per_language,
        "note": "Proposed pilot thresholds; requires customer agreement. Reviews are supplied by humans, not independently authenticated by this tool. Passing is an empirical gate, not a guarantee of perfect accuracy.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", choices=["en", "hi", "ta"], default=["en", "hi", "ta"])
    args = parser.parse_args()
    raw = args.reviews.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    result = assess(rows, tuple(args.languages))
    result["review_file_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS" if result["passed"] else "NOT READY: inspect failed gates in the report")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

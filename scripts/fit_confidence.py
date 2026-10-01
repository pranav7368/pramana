"""Fit train/dev human reviews only; held-out test answers are never training data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from pramana.confidence.fusion import FEATURE_ORDER, ConfidenceModel, expected_calibration_error


def fit_reviews(train: list[dict], dev: list[dict], language: str) -> tuple[ConfidenceModel, dict]:
    seen_qids, seen_parallel, seen_questions = set(), set(), set()
    for split, rows in (("train", train), ("dev", dev)):
        if len(rows) < 20:
            raise ValueError("At least 20 attributed reviews per split are required; more is recommended")
        labels = set()
        for row in rows:
            if row.get("split") != split or row.get("language") != language:
                raise ValueError("Wrong split or language; test data cannot be fitted")
            if row.get("human_reviewed") is not True or not str(row.get("reviewer", "")).strip():
                raise ValueError("Attributed human reviews are required")
            qid, parallel = row.get("qid"), row.get("parallel_id")
            question = " ".join(str(row.get("question", "")).casefold().split())
            if not qid or not parallel or not question:
                raise ValueError("qid, parallel_id and question are required for leakage checks")
            if qid in seen_qids or parallel in seen_parallel or question in seen_questions:
                raise ValueError("Duplicate or overlapping questions across train/dev splits")
            seen_qids.add(qid)
            seen_parallel.add(parallel)
            seen_questions.add(question)
            if type(row.get("fully_supported")) is not bool:
                raise ValueError("fully_supported must be the human boolean label")
            labels.add(row["fully_supported"])
            features = row.get("features")
            if not isinstance(features, dict) or not features or set(features) - set(FEATURE_ORDER):
                raise ValueError("Unknown or missing deployment features")
            if any(type(v) not in {int, float} or not math.isfinite(v) for v in features.values()):
                raise ValueError("Features must be finite numeric values")
        if labels != {False, True}:
            raise ValueError("Both supported and unsupported reviewed examples are required per split")
    model = ConfidenceModel(language=language)
    model.fit([r["features"] for r in train], [int(r["fully_supported"]) for r in train])
    if any(set(model.feature_names) - set(r["features"]) for r in dev):
        raise ValueError("Development features differ from the fitted deployment feature set")
    raw = [model.score(r["features"]).raw_score for r in dev]
    labels = [int(r["fully_supported"]) for r in dev]
    model.calibrator.fit(raw, labels)
    calibrated = [model.score(r["features"]).score for r in dev]
    return model, {"language": language, "train_count": len(train), "dev_count": len(dev),
                   "dev_ece": expected_calibration_error(calibrated, labels),
                   "heldout_test_evaluated": False,
                   "note": "Dev calibration is not a held-out quality result. Review attribution is not identity verification."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--language", choices=["en", "hi", "ta"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    provenance = args.out.with_suffix(".provenance.json")
    if args.out.exists() or provenance.exists():
        parser.error("Choose a fresh artifact path; approved models must not be overwritten")
    contents = [p.read_bytes() for p in (args.train, args.dev)]
    train, dev = [[json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
                  for raw in contents]
    try:
        model, report = fit_reviews(train, dev, args.language)
    except ValueError as exc:
        parser.error(str(exc))
    model.save(args.out)
    report["input_sha256"] = {split: hashlib.sha256(raw).hexdigest()
                              for split, raw in zip(("train", "dev"), contents, strict=True)}
    report["model_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    provenance.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Fitted artifact saved. Held-out human acceptance is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

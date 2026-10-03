#!/usr/bin/env python
"""Check an evaluation question set before any experiment uses it.

    python scripts/validate_dataset.py data/eval/questions.jsonl --corpus data/raw
    python scripts/validate_dataset.py examples/eval/questions.example.jsonl \
        --corpus examples/corpus --pilot

Exit status is 1 when the set has errors (schema, leakage, inconsistent
parallel groups, gold evidence missing from the corpus). Warnings describe
departures from the planned balance and must be reported with any result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pramana.evaluation.dataset import load_questions, validate_questions  # noqa: E402
from pramana.utils.console import safe_text, setup_console  # noqa: E402


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--corpus", type=Path, help="corpus root with {en,hi,ta}/ folders; checks gold chunk ids")
    ap.add_argument("--pilot", action="store_true", help="skip size and balance warnings for a pilot subset")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args()

    try:
        rows = load_questions(args.questions)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    chunk_ids = None
    if args.corpus:
        from pramana.ingestion.corpus import load_corpus

        chunk_ids = [c.chunk_id for chunks in load_corpus(args.corpus).values() for c in chunks]
    report = validate_questions(rows, chunk_ids, full_scale=not args.pilot)
    print(safe_text(report.format()))
    if args.json:
        args.json.write_text(json.dumps({"ok": report.ok, "errors": report.errors,
                                         "warnings": report.warnings, "stats": report.stats},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate an evaluation question set against `docs/05_DATASET_SPEC.md`.

The spec's controls only protect the results if they are checked mechanically
before any experiment runs. Errors make the set unusable; warnings are departures
from the planned design that must be stated when results are reported.

The check that matters most is split leakage across languages: if ``ins-042``
is in *dev* for Hindi and *test* for Tamil, thresholds tuned on dev have already
seen the test question in translation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pramana.schemas import LANGUAGES

ANSWERABILITY_SHARE = {"A": 0.50, "B": 0.20, "C": 0.20, "D": 0.10}
QUESTION_TYPE_SHARE = {
    "factual": 0.30, "numeric": 0.20, "negation": 0.15,
    "multi_hop": 0.15, "procedural": 0.10, "comparative": 0.10,
}
PROVENANCE = {"human_authored", "mt_raw", "mt_verified"}
SPLITS = {"train", "dev", "test"}
SCRIPTS = {"native", "roman"}
REQUIRED = ("qid", "parallel_id", "language", "question", "answerability", "split", "provenance")
MIN_PER_LANGUAGE = 150
SHARE_TOLERANCE = 0.10


@dataclass(slots=True)
class DatasetReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = [f"items: {self.stats.get('items', 0)}"]
        for key in ("by_language", "by_answerability", "by_split", "by_provenance", "roman_by_language"):
            if self.stats.get(key):
                lines.append(f"{key}: {json.dumps(self.stats[key], ensure_ascii=False, sort_keys=True)}")
        lines += [f"ERROR   {e}" for e in self.errors]
        lines += [f"warning {w}" for w in self.warnings]
        lines.append("RESULT: " + ("usable" if self.ok else "not usable until the errors are fixed"))
        return "\n".join(lines)


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {n}: invalid JSON ({exc.msg})") from None
        if not isinstance(row, dict):
            raise ValueError(f"line {n}: expected a JSON object")
        rows.append(row)
    return rows


def validate_questions(
    rows: list[dict[str, Any]], corpus_chunk_ids: Iterable[str] | None = None, *, full_scale: bool = True,
) -> DatasetReport:
    """Check schema, internal consistency, parallel structure and planned balance.

    ``full_scale=False`` skips the size and balance warnings, for pilot subsets.
    """
    report = DatasetReport()
    known_chunks = set(corpus_chunk_ids) if corpus_chunk_ids is not None else None
    seen_qids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, row in enumerate(rows, 1):
        where = f"item {i} ({row.get('qid', '?')})"
        missing = [k for k in REQUIRED if not str(row.get(k, "")).strip()]
        if missing:
            report.errors.append(f"{where}: missing {', '.join(missing)}")
            continue
        if row["qid"] in seen_qids:
            report.errors.append(f"{where}: duplicate qid")
        seen_qids.add(row["qid"])
        groups[row["parallel_id"]].append(row)

        lang, cls = row["language"], row["answerability"]
        if lang not in LANGUAGES:
            report.errors.append(f"{where}: unknown language {lang!r}")
        if cls not in ANSWERABILITY_SHARE:
            report.errors.append(f"{where}: answerability must be A, B, C or D")
        if row["split"] not in SPLITS:
            report.errors.append(f"{where}: split must be one of {sorted(SPLITS)}")
        if row["provenance"] not in PROVENANCE:
            report.errors.append(f"{where}: provenance must be one of {sorted(PROVENANCE)}")
        if row.get("script", "native") not in SCRIPTS:
            report.errors.append(f"{where}: script must be native or roman")
        if row.get("script") == "roman" and lang == "en":
            report.errors.append(f"{where}: English items cannot be Romanised variants")

        expected_abstain = cls in {"C", "D"}
        if "should_abstain" in row and bool(row["should_abstain"]) != expected_abstain:
            report.errors.append(f"{where}: should_abstain contradicts answerability {cls}")
        gold = row.get("gold_chunk_ids") or []
        if not isinstance(gold, list):
            report.errors.append(f"{where}: gold_chunk_ids must be a list")
            gold = []
        if cls in {"A", "B"} and not gold:
            report.errors.append(f"{where}: answerable items need gold_chunk_ids")
        if cls == "C" and gold:
            report.errors.append(f"{where}: out-of-scope items cannot have gold evidence")
        if cls in {"A", "B"} and not str(row.get("gold_answer", "")).strip():
            report.warnings.append(f"{where}: no gold_answer")
        if known_chunks is not None:
            unknown = [c for c in gold if c not in known_chunks]
            if unknown:
                report.errors.append(f"{where}: gold chunk not in corpus: {unknown[:2]}")

        qc = row.get("qc") or {}
        if row["provenance"] == "mt_raw" and row["split"] == "test":
            report.warnings.append(f"{where}: unverified machine translation in the test split")
        if row["provenance"] != "human_authored" and qc.get("human_reviewed") is not True:
            report.warnings.append(f"{where}: translated item without human review")
        if not row.get("dataset_version"):
            report.warnings.append(f"{where}: no dataset_version")

    for pid, members in sorted(groups.items()):
        splits = {m["split"] for m in members}
        if len(splits) > 1:
            report.errors.append(f"parallel_id {pid}: split differs across languages {sorted(splits)} (leakage)")
        for key in ("answerability", "question_type", "domain"):
            values = {m.get(key) for m in members}
            if len(values) > 1:
                report.errors.append(f"parallel_id {pid}: {key} differs across languages {sorted(map(str, values))}")
        native = Counter(m["language"] for m in members if m.get("script", "native") == "native")
        if any(n > 1 for n in native.values()):
            report.errors.append(f"parallel_id {pid}: more than one native item per language")
        absent = [lang for lang in LANGUAGES if lang not in native]
        if absent:
            report.warnings.append(f"parallel_id {pid}: no native item for {', '.join(absent)}")

    valid = [r for r in rows if all(str(r.get(k, "")).strip() for k in REQUIRED)]
    native_rows = [r for r in valid if r.get("script", "native") == "native"]
    report.stats = {
        "items": len(rows),
        "by_language": dict(Counter(r["language"] for r in native_rows)),
        "by_answerability": dict(Counter(r["answerability"] for r in native_rows)),
        "by_split": dict(Counter(r["split"] for r in valid)),
        "by_provenance": dict(Counter(r["provenance"] for r in valid)),
        "roman_by_language": dict(Counter(r["language"] for r in valid if r.get("script") == "roman")),
    }
    if full_scale and native_rows:
        _check_balance(report, native_rows)
    return report


def _check_balance(report: DatasetReport, rows: list[dict[str, Any]]) -> None:
    for lang in LANGUAGES:
        members = [r for r in rows if r["language"] == lang]
        if len(members) < MIN_PER_LANGUAGE:
            report.warnings.append(f"{lang}: {len(members)} questions, below the minimum viable {MIN_PER_LANGUAGE}")
        if not members:
            continue
        for field_name, plan in (("answerability", ANSWERABILITY_SHARE), ("question_type", QUESTION_TYPE_SHARE)):
            counts = Counter(r.get(field_name) for r in members)
            for value, share in plan.items():
                actual = counts.get(value, 0) / len(members)
                if abs(actual - share) > SHARE_TOLERANCE:
                    report.warnings.append(
                        f"{lang}: {field_name} {value} is {actual:.0%} (planned {share:.0%})"
                    )

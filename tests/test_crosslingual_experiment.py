"""Cross-language experiment harness: expectations, scoring and an offline wiring run."""
from __future__ import annotations

import csv
import json

import pytest
from scripts import run_crosslingual as xl

ROW = {"qid": "hi-q1", "parallel_id": "q1", "language": "hi", "question": "?", "answerability": "A",
       "evidence_languages": ["en", "hi"]}


class TestExpectations:
    @pytest.mark.parametrize("answerability,evidence,document,expected", [
        ("A", ["en", "hi"], "en", "answer"),
        ("A", ["en", "hi"], "ta", "abstain"),
        ("B", ["en", "hi", "ta"], "ta", "partial"),
        ("C", [], "en", "abstain"),
        ("A", None, "ta", "answer"),
        ("D", None, "en", "abstain"),
    ])
    def test_expected_behaviour_depends_on_the_document(self, answerability, evidence, document, expected):
        row = {**ROW, "answerability": answerability, "evidence_languages": evidence}
        if evidence is None:
            row.pop("evidence_languages")
        assert xl.expected_behaviour(row, document) == expected

    def test_limit_keeps_whole_parallel_groups(self):
        rows = [{**ROW, "qid": f"{lang}-{pid}", "language": lang, "parallel_id": pid}
                for pid in ("q1", "q2", "q3") for lang in ("en", "hi", "ta")]
        chosen = xl.select_rows(rows, ["en", "hi", "ta"], 2)
        assert {r["parallel_id"] for r in chosen} == {"q1", "q2"} and len(chosen) == 6

    def test_unanswerable_rows_fall_back_to_the_english_reference(self):
        rows = [{"qid": "en-q", "parallel_id": "q", "language": "en", "gold_answer": "21 days"},
                {"qid": "ta-q", "parallel_id": "q", "language": "ta", "gold_answer": ""},
                {"qid": "hi-x", "parallel_id": "x", "language": "hi", "gold_answer": ""}]
        refs = xl.reference_answers(rows)
        assert refs["ta-q"] == "(en reference) 21 days" and refs["hi-x"] == "(should abstain)"


def _record(condition, expected, abstained, supported=1, contradicted=0, n_claims=1, error=None):
    return {"condition": condition, "question_language": "hi", "document_language": "en" if condition == "cross" else "hi",
            "expected": expected, "abstained": abstained, "supported": supported, "contradicted": contradicted,
            "n_claims": n_claims, "error": error, "latency_ms": 10.0}


def test_summary_counts_behaviour_by_condition():
    summary = xl.summarise([
        _record("cross", "answer", False), _record("cross", "answer", True, n_claims=0),
        _record("cross", "abstain", True, n_claims=0), _record("cross", "answer", False, supported=0, contradicted=1),
        _record("native", "answer", False), _record("native", "answer", False, error="Timeout"),
    ])
    cross = summary["condition:cross"]
    assert cross["answered_when_answerable"] == {"count": 2, "of": 3, "rate": 0.6667}
    assert cross["abstained_when_unanswerable"]["rate"] == 1.0
    assert cross["released_with_a_contradicted_claim"] == {"count": 1, "of": 2, "rate": 0.5}
    assert summary["condition:native"]["errors"] == 1
    assert "pair:hi<-en" in summary


def test_score_uses_only_human_marks(tmp_path):
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=xl.REVIEW_FIELDS)
        writer.writeheader()
        for n, (condition, mark) in enumerate([("cross", "correct"), ("cross", "partial"),
                                                ("native", "incorrect"), ("native", "")], 1):
            writer.writerow({**dict.fromkeys(xl.REVIEW_FIELDS, ""), "review_id": n, "condition": condition,
                             "question_language": "hi", "document_language": "en", "human_mark": mark})
    result = xl.score(path)
    assert result["marked"] == 3 and result["unmarked"] == 1
    assert result["groups"]["condition:cross"]["accuracy"] == 0.75
    assert result["groups"]["condition:native"]["n"] == 1


def test_score_rejects_unknown_marks(tmp_path):
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=xl.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerow({**dict.fromkeys(xl.REVIEW_FIELDS, ""), "review_id": 1, "human_mark": "maybe"})
    with pytest.raises(SystemExit, match="human_mark"):
        xl.score(path)


def test_offline_run_writes_every_artifact(tmp_path):
    assert xl.main(["--offline", "--limit", "1", "--out", str(tmp_path)]) == 0
    raw = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(raw) == 4 * 3  # en, hi, hi-roman and ta questions, each against three documents
    assert {r["condition"] for r in raw} == {"native", "cross"}
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["offline"] is True
    with (tmp_path / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 12

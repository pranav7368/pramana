"""The evaluation set's controls are checked mechanically, not by discipline."""

from __future__ import annotations

from pathlib import Path

from pramana.evaluation.dataset import load_questions, validate_questions

ROOT = Path(__file__).resolve().parents[1]


def _item(qid, pid, lang, *, cls="A", split="dev", gold=("c1",), **extra):
    return {"qid": qid, "parallel_id": pid, "language": lang, "question": f"q {qid}",
            "answerability": cls, "split": split, "provenance": "human_authored",
            "gold_chunk_ids": list(gold) if cls in "AB" else [], "gold_answer": "a",
            "question_type": "factual", "domain": "insurance", "dataset_version": "t", **extra}


def _parallel(pid, **kw):
    return [_item(f"{lang}-{pid}", pid, lang, **kw) for lang in ("en", "hi", "ta")]


def test_a_clean_parallel_group_has_no_errors():
    report = validate_questions(_parallel("p1"), full_scale=False)
    assert report.ok and not report.warnings


def test_split_leakage_across_languages_is_an_error():
    rows = _parallel("p1")
    rows[2]["split"] = "test"
    assert any("leakage" in e for e in validate_questions(rows, full_scale=False).errors)


def test_answerability_must_agree_across_a_parallel_group():
    rows = _parallel("p1")
    rows[1].update(answerability="C", gold_chunk_ids=[])
    assert any("answerability differs" in e for e in validate_questions(rows, full_scale=False).errors)


def test_abstention_label_must_follow_the_class():
    rows = _parallel("p1", cls="D", should_abstain=False)
    assert any("should_abstain" in e for e in validate_questions(rows, full_scale=False).errors)


def test_gold_evidence_rules():
    no_gold = _parallel("p1", gold=())
    assert any("need gold_chunk_ids" in e for e in validate_questions(no_gold, full_scale=False).errors)
    missing = validate_questions(_parallel("p2"), corpus_chunk_ids=["other"], full_scale=False)
    assert any("not in corpus" in e for e in missing.errors)


def test_duplicates_and_bad_enums_are_errors():
    rows = [*_parallel("p1"), _item("en-p1", "p9", "en", split="holdout")]
    errors = validate_questions(rows, full_scale=False).errors
    assert any("duplicate qid" in e for e in errors)
    assert any("split must be" in e for e in errors)


def test_unreviewed_translation_in_test_is_flagged():
    rows = _parallel("p1", split="test")
    rows[1].update(provenance="mt_raw", qc={"human_reviewed": False})
    warnings = validate_questions(rows, full_scale=False).warnings
    assert any("unverified machine translation in the test split" in w for w in warnings)


def test_full_scale_reports_size_and_balance():
    warnings = validate_questions(_parallel("p1")).warnings
    assert any("below the minimum viable" in w for w in warnings)
    assert any("answerability C is 0%" in w for w in warnings)


def test_bundled_example_is_valid_against_the_bundled_corpus():
    from pramana.ingestion.corpus import load_corpus

    rows = load_questions(ROOT / "examples/eval/questions.example.jsonl")
    chunks = [c.chunk_id for cs in load_corpus(ROOT / "examples/corpus").values() for c in cs]
    report = validate_questions(rows, chunks, full_scale=False)
    assert report.ok, report.errors
    assert report.stats["roman_by_language"] == {"hi": 1}

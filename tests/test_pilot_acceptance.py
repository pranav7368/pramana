from dataclasses import replace

import pytest
from scripts.pilot_acceptance import assess
from scripts.run_experiment import Item, Record, analyse, print_report, run_system

from pramana.evaluation.metrics import LanguageBreakdown


def review(i, **changes):
    return dict(qid=f"q{i}", language="en", split="test", human_reviewed=True, reviewer="reviewer-1",
                abstained=i < 10, should_abstain=i < 10, fully_supported=True, confidence=1.0,
                confidence_calibrated=True, latency_ms=100, baseline_latency_ms=100, **changes)


def test_empty_or_one_language_reviews_cannot_pass_a_three_language_pilot():
    assert not assess([])["passed"]
    assert not assess([review(i) for i in range(50)])["passed"]


def test_gate_requires_real_reviews_calibration_and_coverage():
    rows = [review(i) for i in range(50)]
    assert assess(rows, ("en",))["passed"]
    for row in rows:
        row["confidence_calibrated"] = False
    assert not assess(rows, ("en",))["passed"]
    rows[0]["human_reviewed"] = False
    with pytest.raises(ValueError, match="human reviews"):
        assess(rows, ("en",))


def test_abstaining_on_everything_fails_acceptance():
    rows = [review(i) for i in range(50)]
    for row in rows:
        row["abstained"] = True
    report = assess(rows, ("en",))
    assert not report["passed"]
    assert report["per_language"]["en"]["unsupported_answer_rate"] is None


def test_duplicate_reviews_and_test_leakage_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        assess([review(1), review(1)], ("en",))
    row = review(1)
    row["split"] = "dev"
    with pytest.raises(ValueError, match="held-out"):
        assess([row], ("en",))


def test_metric_direction_and_better_languages():
    worse = LanguageBreakdown("hallucination_rate", {"en": 0.1, "hi": 0.2, "ta": 0.05}, higher_is_better=False)
    assert worse.worst_gap[0] == "hi" and not worse.meets_criterion()
    better = LanguageBreakdown("f1", {"en": 0.8, "hi": 0.95, "ta": 0.95})
    assert better.meets_criterion()
    zero = LanguageBreakdown("hallucination_rate", {"en": 0.0, "hi": 0.1}, higher_is_better=False)
    assert not zero.meets_criterion()


def test_failed_experiment_is_reported_instead_of_disappearing(capsys):
    class Broken:
        def run(self, *args, **kwargs):
            raise RuntimeError("private provider error")
    records = run_system("pramana", Broken(), [Item("q", "q", "en", "question")])
    assert len(records) == 1 and records[0].error == "RuntimeError"
    result = analyse(records)
    assert result["systems"]["pramana"]["per_language"]["en"]["failures"] == 1
    print_report(result)
    assert "private provider error" not in capsys.readouterr().out


def test_regression_rate_uses_attempts_and_failed_pairs_are_excluded():
    base = Record("vanilla", "q", "q", "en", "A", "answer", 0.8, "HIGH", 1, 0.5,
                  1, 0, 0, False, False, False, 0, [], ["retrieved"], [], 100)
    corrected = replace(base, system="pramana", correction_attempts=2, correction_regressions=1)
    failed = replace(corrected, qid="failed", parallel_id="failed", error="ProviderError")
    report = analyse([base, corrected, failed])
    assert report["systems"]["pramana"]["per_language"]["en"]["regression_rate"] == 0.5
    assert report["comparisons"]["pramana_vs_vanilla"]["n"] == 1
    assert report["systems"]["pramana"]["cross_lingual"]["meets_10pct_criterion"] is None

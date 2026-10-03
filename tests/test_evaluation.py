"""Tests for the evaluation metrics.

A bug here is the most dangerous kind in this project: it produces a plausible
number in a results table that nobody can tell is wrong. Each test therefore
checks a value that can be computed by hand.
"""

from __future__ import annotations

import pytest

from pramana.evaluation.metrics import (
    LanguageBreakdown,
    abstention_metrics,
    bootstrap_ci,
    classification_report,
    correction_metrics,
    holm_bonferroni,
    paired_bootstrap,
    retrieval_metrics,
)

S, C, U = "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"


class TestClassificationReport:
    def test_perfect_prediction(self):
        r = classification_report([S, C, U], [S, C, U])
        assert r.macro_f1 == pytest.approx(1.0)
        assert r.accuracy == pytest.approx(1.0)

    def test_hand_computable_case(self):
        gold = [S, S, S, C]
        pred = [S, S, C, C]
        r = classification_report(gold, pred)
        # SUPPORTED: tp=2 fp=0 fn=1 -> P=1.0 R=0.667 F1=0.8
        assert r.per_class[S].precision == pytest.approx(1.0)
        assert r.per_class[S].recall == pytest.approx(2 / 3)
        assert r.per_class[S].f1 == pytest.approx(0.8)
        # CONTRADICTED: tp=1 fp=1 fn=0 -> P=0.5 R=1.0 F1=0.667
        assert r.per_class[C].precision == pytest.approx(0.5)
        assert r.per_class[C].f1 == pytest.approx(2 / 3)

    def test_macro_f1_punishes_ignoring_a_minority_class(self):
        """Accuracy would reward a detector that flags nothing, because SUPPORTED
        dominates the label distribution. Macro-F1 is the headline for that reason."""
        gold = [S] * 9 + [C]
        pred = [S] * 10  # never predicts CONTRADICTED
        r = classification_report(gold, pred)
        assert r.accuracy == pytest.approx(0.9)
        assert r.macro_f1 < 0.5, "macro-F1 must expose the ignored class"

    def test_binary_f1_collapses_to_supported_vs_not(self):
        r = classification_report([S, C, U, S], [S, U, U, S])
        assert r.binary_f1() == pytest.approx(1.0)

    def test_confusion_matrix_records_the_critical_cell(self):
        """Confusing CONTRADICTED with UNVERIFIABLE sends the wrong correction: one
        should be fixed, the other removed."""
        r = classification_report([C, C, U], [U, C, U])
        assert r.confusion[(C, U)] == 1
        assert "CONTR" in r.format_confusion()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="3 gold labels but 2 predictions"):
            classification_report([S, C, U], [S, C])


class TestRetrievalMetrics:
    def test_gold_at_rank_one(self):
        m = retrieval_metrics([["a", "b", "c"]], [["a"]], k=3)
        assert m.recall_at_k == pytest.approx(1.0)
        assert m.mrr == pytest.approx(1.0)
        assert m.ndcg_at_k == pytest.approx(1.0)

    def test_gold_at_rank_three_scores_lower(self):
        top = retrieval_metrics([["a", "b", "c"]], [["a"]], k=3)
        deep = retrieval_metrics([["b", "c", "a"]], [["a"]], k=3)
        assert deep.mrr == pytest.approx(1 / 3)
        assert deep.ndcg_at_k < top.ndcg_at_k

    def test_gold_outside_k_is_a_miss(self):
        m = retrieval_metrics([["b", "c", "d", "a"]], [["a"]], k=3)
        assert m.recall_at_k == 0.0 and m.mrr == 0.0

    def test_empty_rate_is_tracked(self):
        """Should correlate with the deliberately unanswerable questions."""
        m = retrieval_metrics([[], ["a"]], [["x"], ["a"]], k=3)
        assert m.empty_rate == pytest.approx(0.5)

    def test_partial_recall_of_multiple_gold_chunks(self):
        m = retrieval_metrics([["a", "z"]], [["a", "b"]], k=5)
        assert m.recall_at_k == pytest.approx(0.5)


class TestAbstentionMetrics:
    def test_perfect_abstention(self):
        m = abstention_metrics([True, False], [True, False])
        assert m.precision == 1.0 and m.recall == 1.0
        assert m.over_abstention_rate == 0.0

    def test_over_abstention_is_caught(self):
        """The failure a hallucination-only view is blind to: a system that never
        answers scores a perfect hallucination rate and is useless."""
        m = abstention_metrics([True, True, True], [False, False, True])
        assert m.over_abstention_rate == pytest.approx(1.0)
        assert m.precision == pytest.approx(1 / 3)

    def test_never_abstaining_scores_zero_recall(self):
        m = abstention_metrics([False, False], [True, True])
        assert m.recall == 0.0 and m.rate == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="decisions but"):
            abstention_metrics([True], [True, False])


class TestCorrectionMetrics:
    def test_improvement_is_measured(self):
        m = correction_metrics(before=[0.5, 0.4], after=[0.1, 0.0], regressed=[False, False])
        assert m.relative_reduction == pytest.approx(8 / 9)
        assert m.improvement_rate == pytest.approx(1.0)
        assert m.regression_rate == 0.0

    def test_regression_is_reported_not_hidden(self):
        """A stage that improves 60% and degrades 20% is not the same as one that
        improves 45% and degrades none."""
        m = correction_metrics(
            before=[0.5, 0.0, 0.5, 0.5, 0.5],
            after=[0.1, 0.5, 0.1, 0.1, 0.1],
            regressed=[False, True, False, False, False],
        )
        assert m.regression_rate == pytest.approx(0.2)
        assert m.improvement_rate == pytest.approx(0.8)
        assert m.net_improvement == pytest.approx(0.6)
        assert "REGRESSED" in str(m)

    def test_relevance_retention_catches_a_bad_trade(self):
        """Faithfulness alone is gameable -- "I cannot answer" is perfectly
        faithful and useless."""
        m = correction_metrics(
            before=[0.5], after=[0.0], regressed=[False],
            relevance_before=[5.0], relevance_after=[2.0],
        )
        assert m.relevance_retention == pytest.approx(0.4)

    def test_empty_input_is_safe(self):
        assert correction_metrics([], [], []).n_attempts == 0


class TestUncertainty:
    def test_interval_brackets_the_point_estimate(self):
        ci = bootstrap_ci([0.8, 0.82, 0.79, 0.81, 0.83], resamples=2000)
        assert ci.lower <= ci.point <= ci.upper

    def test_high_variance_widens_the_interval(self):
        tight = bootstrap_ci([0.5] * 20, resamples=2000)
        loose = bootstrap_ci([0.0, 1.0] * 10, resamples=2000)
        assert loose.width > tight.width

    def test_intervals_are_reproducible(self):
        """A reported interval must be regenerable, so the seed is fixed."""
        a = bootstrap_ci([0.1, 0.5, 0.9], resamples=1000)
        b = bootstrap_ci([0.1, 0.5, 0.9], resamples=1000)
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_empty_input_is_safe(self):
        assert bootstrap_ci([]).point == 0.0


class TestPairedBootstrap:
    def test_detects_a_consistent_difference(self):
        better = [0.9] * 30
        worse = [0.5] * 30
        r = paired_bootstrap(better, worse, resamples=2000)
        assert r.difference == pytest.approx(0.4)
        assert r.significant

    def test_identical_systems_are_not_significant(self):
        r = paired_bootstrap([0.5] * 20, [0.5] * 20, resamples=2000)
        assert r.difference == pytest.approx(0.0)
        assert not r.significant

    def test_noisy_small_difference_is_not_significant(self):
        """With a few hundred items the interval is often wide enough to change the
        conclusion. Reporting the point estimate alone would hide that."""
        a = [0.5, 0.9, 0.1, 0.6, 0.4, 0.8, 0.2]
        b = [0.4, 0.9, 0.2, 0.5, 0.5, 0.7, 0.3]
        assert not paired_bootstrap(a, b, resamples=3000).significant

    def test_unpaired_input_raises(self):
        with pytest.raises(ValueError, match="must be paired"):
            paired_bootstrap([0.1, 0.2], [0.1])


class TestHolmBonferroni:
    def test_clear_winner_survives(self):
        out = holm_bonferroni({"a": 0.001, "b": 0.9, "c": 0.8})
        assert out["a"] and not out["b"]

    def test_marginal_results_do_not_survive_correction(self):
        """Comparing against six baselines makes a spurious win likely by chance."""
        out = holm_bonferroni({f"b{i}": 0.04 for i in range(6)})
        assert not any(out.values())


class TestLanguageBreakdown:
    def test_gap_is_computed_against_english(self):
        b = LanguageBreakdown(metric="macro_f1", by_language={"en": 0.80, "hi": 0.76, "ta": 0.60})
        assert b.gap("hi") == pytest.approx(-0.05)
        assert b.gap("ta") == pytest.approx(-0.25)

    def test_worst_gap_is_surfaced(self):
        b = LanguageBreakdown(metric="macro_f1", by_language={"en": 0.80, "hi": 0.78, "ta": 0.55})
        lang, gap = b.worst_gap
        assert lang == "ta" and gap < -0.3

    def test_success_criterion_is_checked(self):
        """01_PROBLEM_STATEMENT.md §4 requires Hindi and Tamil within 10% relative
        of English."""
        ok = LanguageBreakdown(metric="f1", by_language={"en": 0.80, "hi": 0.76, "ta": 0.74})
        bad = LanguageBreakdown(metric="f1", by_language={"en": 0.80, "hi": 0.76, "ta": 0.60})
        assert ok.meets_criterion()
        assert not bad.meets_criterion()

    def test_format_flags_a_failing_gap(self):
        b = LanguageBreakdown(metric="f1", by_language={"en": 0.80, "ta": 0.50})
        assert "EXCEEDS" in b.format()

    def test_single_language_has_no_gap(self):
        b = LanguageBreakdown(metric="f1", by_language={"en": 0.80})
        assert b.worst_gap == ("en", 0.0)

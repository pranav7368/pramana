"""Tests for confidence fusion (stage 4) and the correction policy (stage 5).

The failures guarded against here are silent by nature: a miscalibrated score and
a correction that degrades an answer both look like success from the outside.
"""

from __future__ import annotations

import pytest

from pramana.confidence.fusion import (
    FEATURE_ORDER,
    ConfidenceModel,
    TemperatureCalibrator,
    brier_score,
    collect_features,
    expected_calibration_error,
    reliability_bins,
)
from pramana.correction.policy import (
    CorrectionExecutor,
    CorrectionPolicy,
    RegressionTracker,
    VerdictProfile,
    build_correction_messages,
    prune_claims,
)
from pramana.generation.base import Completion, GenerationResponse
from pramana.schemas import (
    Action,
    Chunk,
    Claim,
    ClaimVerdict,
    DetectionResult,
    Draft,
    RetrievalResult,
    RetrievedChunk,
    Verdict,
)


def detection(*verdicts: Verdict, spans=None) -> DetectionResult:
    spans = spans or [None] * len(verdicts)
    return DetectionResult(
        [
            ClaimVerdict(
                claim=Claim(f"c{i}", f"claim {i} text here", "en", source_span=spans[i]),
                verdict=v,
                entailment_prob=0.8 if v is Verdict.SUPPORTED else 0.1,
                contradiction_prob=0.8 if v is Verdict.CONTRADICTED else 0.1,
                neutral_prob=0.8 if v is Verdict.UNVERIFIABLE else 0.1,
            )
            for i, v in enumerate(verdicts)
        ]
    )


def retrieval(top: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        normalized_query="q",
        language="en",
        script="roman",
        chunks=[
            RetrievedChunk(chunk=Chunk("c0", "d", "evidence text", "en"), rank=0, fused_score=top)
        ],
    )


class _Provider:
    name = "p"

    def __init__(self, *replies: str):
        self.queue = list(replies)
        self.prompts: list[str] = []

    def generate(self, request):
        self.prompts.append("\n".join(m.content for m in request.messages))
        return GenerationResponse(
            completions=[Completion(self.queue.pop(0) if self.queue else "")],
            model="m",
            provider="p",
        )

    def capabilities(self, model):
        return set()

    def available_models(self):
        return ["m"]

    def health_check(self):
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — confidence
# ══════════════════════════════════════════════════════════════════════════════


class TestFeatureCollection:
    def test_gathers_all_available_signals(self):
        f = collect_features(
            retrieval(),
            Draft(text="a long enough answer here", language="en", samples=["a b", "a c"]),
            detection(Verdict.SUPPORTED, Verdict.CONTRADICTED),
        )
        assert {"s1_top_score", "s2_supported_ratio", "s4_self_consistency"} <= set(f)

    def test_s3_absent_when_no_logprobs(self):
        """S3 must be *missing*, not zero: a zero is a real value the model would
        learn from, corrupting calibration silently."""
        f = collect_features(retrieval(), Draft(text="x y z", language="en"), detection())
        assert "s3_mean_logprob" not in f

    def test_s3_present_when_logprobs_available(self):
        f = collect_features(
            retrieval(), Draft(text="x y z", language="en", mean_token_logprob=-0.4), detection()
        )
        assert f["s3_mean_logprob"] == pytest.approx(-0.4)

    def test_every_collected_feature_is_declared(self):
        """A feature absent from FEATURE_ORDER is silently ignored at fit time."""
        f = collect_features(
            retrieval(),
            Draft(text="a b c d", language="en", samples=["a b"], mean_token_logprob=-0.3),
            detection(Verdict.SUPPORTED),
        )
        assert set(f) <= set(FEATURE_ORDER), f"undeclared: {set(f) - set(FEATURE_ORDER)}"


class TestCalibration:
    def test_temperature_one_is_identity(self):
        assert TemperatureCalibrator(1.0).apply(0.7) == pytest.approx(0.7, abs=1e-6)

    def test_higher_temperature_softens_confidence(self):
        assert TemperatureCalibrator(3.0).apply(0.95) < 0.95

    def test_fitting_reduces_calibration_error(self):
        """Overconfident scores are the documented RAG failure -- calibration must
        actually pull them back."""
        probs = [0.95] * 10 + [0.9] * 10
        labels = [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5  # only ~50% correct
        before = expected_calibration_error(probs, labels)

        cal = TemperatureCalibrator()
        cal.fit(probs, labels)
        after = expected_calibration_error([cal.apply(p) for p in probs], labels)

        assert after < before
        assert cal.temperature > 1.0, "should have softened, not sharpened"

    def test_fit_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="3 probabilities but 2 labels"):
            TemperatureCalibrator().fit([0.1, 0.2, 0.3], [1, 0])

    def test_fit_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            TemperatureCalibrator().fit([], [])


class TestCalibrationMetrics:
    def test_perfect_calibration_scores_zero_ece(self):
        assert expected_calibration_error([1.0] * 5 + [0.0] * 5, [1] * 5 + [0] * 5) == pytest.approx(0.0)

    def test_overconfidence_is_penalised(self):
        assert expected_calibration_error([0.95] * 10, [1] * 5 + [0] * 5) > 0.4

    def test_brier_rewards_accuracy(self):
        assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
        assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)

    def test_reliability_bins_cover_the_unit_interval(self):
        bins = reliability_bins([0.1, 0.5, 0.9], [0, 1, 1], bins=10)
        assert len(bins) == 10
        assert sum(b["count"] for b in bins) == 3

    def test_ece_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="probabilities but"):
            expected_calibration_error([0.5], [1, 0])


class TestConfidenceModel:
    def test_untrained_model_is_marked_uncalibrated(self):
        """An untrained heuristic must never be mistaken for a calibrated score."""
        report = ConfidenceModel().score(
            collect_features(retrieval(), Draft(text="a b c d", language="en"), detection(Verdict.SUPPORTED))
        )
        assert report.calibrator == "none"
        assert any("UNTRAINED" in m for m in report.missing_signals)

    def test_contradiction_drives_confidence_down(self):
        m = ConfidenceModel()
        good = m.score(collect_features(retrieval(), Draft(text="a b c d", language="en"), detection(Verdict.SUPPORTED, Verdict.SUPPORTED)))
        bad = m.score(collect_features(retrieval(), Draft(text="a b c d", language="en"), detection(Verdict.CONTRADICTED, Verdict.SUPPORTED)))
        assert bad.score < good.score

    def test_bands_partition_the_score_range(self):
        m = ConfidenceModel(bands=(0.75, 0.45))
        assert m.band_for(0.9) == "HIGH"
        assert m.band_for(0.6) == "MEDIUM"
        assert m.band_for(0.2) == "LOW"

    def test_fit_learns_a_separating_direction(self):
        rows, labels = [], []
        for i in range(40):
            faithful = i % 2 == 0
            rows.append(
                {
                    "s2_supported_ratio": 1.0 if faithful else 0.2,
                    "s2_contradicted_ratio": 0.0 if faithful else 0.6,
                    "s1_top_score": 0.9 if faithful else 0.2,
                }
            )
            labels.append(1 if faithful else 0)

        m = ConfidenceModel()
        m.fit(rows, labels, epochs=300)

        high = m.score({"s2_supported_ratio": 1.0, "s2_contradicted_ratio": 0.0, "s1_top_score": 0.9})
        low = m.score({"s2_supported_ratio": 0.2, "s2_contradicted_ratio": 0.6, "s1_top_score": 0.2})
        assert high.score > low.score

    def test_fit_drops_features_missing_from_any_row(self):
        """S3 absent on even one row must remove it from the model rather than be
        filled in."""
        rows = [
            {"s2_supported_ratio": 1.0, "s3_mean_logprob": -0.2},
            {"s2_supported_ratio": 0.0},  # no logprobs from this provider
        ]
        m = ConfidenceModel()
        m.fit(rows, [1, 0], epochs=10)
        assert "s3_mean_logprob" not in m.feature_names
        assert "s3_mean_logprob" in m.missing_features

    def test_fit_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="2 rows but 1 labels"):
            ConfidenceModel().fit([{"a": 1.0}, {"a": 2.0}], [1])

    def test_fit_rejects_rows_with_no_common_feature(self):
        with pytest.raises(ValueError, match="no feature is present in every row"):
            ConfidenceModel().fit([{"s1_top_score": 1.0}, {"s2_n_claims": 1.0}], [1, 0])

    def test_importance_is_reportable(self):
        """Which signal matters most is a research finding, and the answer to a
        compliance reviewer asking why a score was low."""
        m = ConfidenceModel()
        m.fit(
            [{"s2_supported_ratio": 1.0, "s1_top_score": 0.5}] * 10
            + [{"s2_supported_ratio": 0.0, "s1_top_score": 0.5}] * 10,
            [1] * 10 + [0] * 10,
            epochs=200,
        )
        ranked = m.importance()
        assert ranked[0][0] == "s2_supported_ratio", "entailment should dominate here"

    def test_round_trip_through_disk(self, tmp_path):
        m = ConfidenceModel()
        m.fit([{"s2_supported_ratio": 1.0}] * 5 + [{"s2_supported_ratio": 0.0}] * 5, [1] * 5 + [0] * 5, epochs=50)
        m.language = "ta"
        path = tmp_path / "model.json"
        m.save(path)

        loaded = ConfidenceModel.load(path)
        assert loaded.language == "ta"
        assert loaded.feature_names == m.feature_names
        features = {"s2_supported_ratio": 1.0}
        assert loaded.score(features).score == pytest.approx(m.score(features).score)

    def test_load_rejects_unknown_features(self, tmp_path):
        """A model fitted against a different feature set would score nonsense."""
        path = tmp_path / "bad.json"
        path.write_text('{"weights": {"mystery": 1.0}, "bias": 0.0, "feature_names": ["mystery"]}')
        with pytest.raises(ValueError, match="unknown features"):
            ConfidenceModel.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — correction
# ══════════════════════════════════════════════════════════════════════════════


class TestVerdictProfile:
    def test_fewer_contradictions_wins_first(self):
        assert VerdictProfile(1, 0, 2).is_better_than(VerdictProfile(3, 1, 0))

    def test_then_fewer_unverifiable(self):
        assert VerdictProfile(2, 0, 1).is_better_than(VerdictProfile(2, 0, 3))

    def test_then_more_supported(self):
        assert VerdictProfile(3, 0, 1).is_better_than(VerdictProfile(2, 0, 1))

    def test_identical_profiles_are_not_better(self):
        """Rollback requires *strict* improvement -- otherwise a no-op revision
        would be accepted and the loop could churn."""
        p = VerdictProfile(2, 1, 1)
        assert not p.is_better_than(p)


class TestPolicyDecisions:
    def policy(self) -> CorrectionPolicy:
        return CorrectionPolicy()

    def test_all_supported_is_accepted(self):
        a = self.policy().decide(detection(Verdict.SUPPORTED, Verdict.SUPPORTED), 0.9, "HIGH", retrieval(), 0)
        assert a is Action.ACCEPT

    def test_contradiction_with_good_retrieval_regenerates(self):
        """The evidence exists and disagrees, so it says what the right answer is."""
        a = self.policy().decide(detection(Verdict.CONTRADICTED, Verdict.SUPPORTED), 0.5, "MEDIUM", retrieval(0.8), 0)
        assert a is Action.REGENERATE

    def test_unverifiable_with_weak_retrieval_re_retrieves(self):
        """Nothing relevant was found -- the retrieval step is the problem, not
        the generator."""
        a = self.policy().decide(detection(Verdict.UNVERIFIABLE, Verdict.SUPPORTED), 0.5, "MEDIUM", retrieval(0.05), 0)
        assert a is Action.RE_RETRIEVE

    def test_minority_unverifiable_is_pruned(self):
        a = self.policy().decide(
            detection(Verdict.SUPPORTED, Verdict.SUPPORTED, Verdict.SUPPORTED, Verdict.UNVERIFIABLE),
            0.6, "MEDIUM", retrieval(0.8), 0,
        )
        assert a is Action.PRUNE

    def test_no_supported_claims_abstains(self):
        a = self.policy().decide(detection(Verdict.UNVERIFIABLE, Verdict.UNVERIFIABLE), 0.4, "LOW", retrieval(), 0)
        assert a is Action.ABSTAIN

    def test_very_low_confidence_abstains(self):
        a = self.policy().decide(detection(Verdict.SUPPORTED), 0.1, "LOW", retrieval(), 0)
        assert a is Action.ABSTAIN

    def test_loop_is_bounded(self):
        """Cost must be provably bounded, not emergent."""
        p = CorrectionPolicy(max_iterations=2)
        a = p.decide(detection(Verdict.CONTRADICTED, Verdict.SUPPORTED), 0.5, "MEDIUM", retrieval(0.8), 2)
        assert a is not Action.REGENERATE, "must stop attempting after the cap"

    def test_empty_detection_accepts(self):
        assert self.policy().decide(DetectionResult([]), 0.5, "MEDIUM", retrieval(), 0) is Action.ACCEPT

    def test_disabled_action_falls_back_safely(self):
        """Ablations disable individual branches; the engine must still terminate."""
        p = CorrectionPolicy(enabled_actions=frozenset({Action.ACCEPT, Action.ABSTAIN}))
        a = p.decide(detection(Verdict.CONTRADICTED, Verdict.SUPPORTED), 0.5, "MEDIUM", retrieval(0.8), 0)
        assert a in (Action.ACCEPT, Action.ABSTAIN)

    def test_decisions_are_deterministic(self):
        p, d, r = self.policy(), detection(Verdict.CONTRADICTED, Verdict.SUPPORTED), retrieval(0.8)
        assert {p.decide(d, 0.5, "MEDIUM", r, 0) for _ in range(5)} == {Action.REGENERATE}


class TestPruning:
    def test_removes_only_the_flagged_span(self):
        text = "The deadline is 30 days. Dental cover is included."
        det = DetectionResult(
            [
                ClaimVerdict(claim=Claim("c0", "The deadline is 30 days.", "en", (0, 24)), verdict=Verdict.SUPPORTED),
                ClaimVerdict(claim=Claim("c1", "Dental cover is included.", "en", (25, 50)), verdict=Verdict.UNVERIFIABLE),
            ]
        )
        out = prune_claims(text, det)
        assert "30 days" in out and "Dental" not in out

    def test_claims_without_spans_are_left_alone(self):
        """A rewritten claim has no span; guessing one would delete the wrong text."""
        text = "The deadline is 30 days."
        det = DetectionResult(
            [ClaimVerdict(claim=Claim("c0", "something else", "en", None), verdict=Verdict.UNVERIFIABLE)]
        )
        assert prune_claims(text, det) == text

    def test_multiple_spans_are_removed_correctly(self):
        """Right-to-left removal keeps earlier offsets valid."""
        text = "AAAA. BBBB. CCCC."
        det = DetectionResult(
            [
                ClaimVerdict(claim=Claim("c0", "AAAA.", "en", (0, 5)), verdict=Verdict.UNVERIFIABLE),
                ClaimVerdict(claim=Claim("c1", "BBBB.", "en", (6, 11)), verdict=Verdict.SUPPORTED),
                ClaimVerdict(claim=Claim("c2", "CCCC.", "en", (12, 17)), verdict=Verdict.UNVERIFIABLE),
            ]
        )
        out = prune_claims(text, det)
        assert "BBBB" in out and "AAAA" not in out and "CCCC" not in out


class TestCorrectionExecutor:
    def executor(self, *replies: str) -> CorrectionExecutor:
        return CorrectionExecutor(provider=_Provider(*replies))

    def test_accept_returns_the_draft_unchanged(self):
        d = Draft(text="original answer", language="en")
        out = self.executor().apply(
            Action.ACCEPT, d, detection(Verdict.SUPPORTED), retrieval(), lambda t: detection(), "en"
        )
        assert out.text == "original answer" and out.accepted and not out.regressed

    def test_abstain_returns_a_message_not_the_answer(self):
        d = Draft(text="a confident wrong answer", language="en")
        out = self.executor().apply(
            Action.ABSTAIN, d, detection(Verdict.UNVERIFIABLE), retrieval(), lambda t: detection(), "en"
        )
        assert "original" not in out.text and out.accepted

    def test_improved_revision_is_accepted(self):
        d = Draft(text="deadline is 15 days", language="en")
        before = detection(Verdict.CONTRADICTED)
        out = self.executor("deadline is 30 days").apply(
            Action.REGENERATE, d, before, retrieval(), lambda t: detection(Verdict.SUPPORTED), "en"
        )
        assert out.accepted and not out.regressed
        assert out.text == "deadline is 30 days"

    def test_non_improving_revision_is_rolled_back(self):
        """The documented risk: self-correction can make a correct answer worse.
        A revision that does not strictly improve is rejected."""
        d = Draft(text="deadline is 30 days", language="en")
        before = detection(Verdict.SUPPORTED)
        out = self.executor("deadline might be 45 days").apply(
            Action.REGENERATE, d, before, retrieval(), lambda t: detection(Verdict.CONTRADICTED), "en"
        )
        assert not out.accepted and out.regressed

    def test_equal_profile_is_rolled_back(self):
        """Strict improvement is required, so a lateral rewrite is rejected."""
        d = Draft(text="original", language="en")
        out = self.executor("a different wording").apply(
            Action.REGENERATE, d, detection(Verdict.UNVERIFIABLE), retrieval(),
            lambda t: detection(Verdict.UNVERIFIABLE), "en",
        )
        assert not out.accepted and not out.regressed

    def test_provider_failure_keeps_the_original(self):
        class _Broken(_Provider):
            def generate(self, request):
                raise RuntimeError("down")

        out = CorrectionExecutor(provider=_Broken()).apply(
            Action.REGENERATE, Draft(text="original", language="en"), detection(Verdict.CONTRADICTED),
            retrieval(), lambda t: detection(Verdict.SUPPORTED), "en",
        )
        assert out.text == "original" and not out.accepted

    def test_pruning_everything_becomes_abstention(self):
        text = "Dental cover is included."
        det = DetectionResult(
            [ClaimVerdict(claim=Claim("c0", text, "en", (0, len(text))), verdict=Verdict.UNVERIFIABLE)]
        )
        out = self.executor().apply(
            Action.PRUNE, Draft(text=text, language="en"), det, retrieval(), lambda t: detection(), "en"
        )
        assert out.action is Action.ABSTAIN


class TestCorrectionPrompt:
    def test_prompt_names_the_specific_failed_claims(self):
        """Evidence-guided correction, not blanket self-critique -- the difference
        that makes correction safe."""
        det = DetectionResult(
            [
                ClaimVerdict(claim=Claim("c0", "The deadline is 15 days.", "en"), verdict=Verdict.CONTRADICTED),
                ClaimVerdict(claim=Claim("c1", "Appeals take 60 days.", "en"), verdict=Verdict.SUPPORTED),
            ]
        )
        msgs = build_correction_messages(Draft(text="answer", language="en"), det, retrieval(), "en")
        joined = "\n".join(m.content for m in msgs)
        assert "The deadline is 15 days." in joined
        assert "Appeals take 60 days." not in joined, "correct claims must not be flagged"

    @pytest.mark.parametrize("language", ["en", "hi", "ta"])
    def test_every_language_has_a_correction_prompt(self, language):
        msgs = build_correction_messages(
            Draft(text="answer", language=language), detection(Verdict.CONTRADICTED), retrieval(), language
        )
        assert msgs[0].content.strip()


class TestRegressionTracker:
    def test_tracks_improvements_and_regressions(self):
        t = RegressionTracker()
        for regressed in (False, False, True):
            t.record(
                CorrectionExecutor(provider=_Provider())._outcome(
                    "t", Action.REGENERATE, VerdictProfile(1, 1, 0), VerdictProfile(2, 0, 0),
                    accepted=not regressed, regressed=regressed, started=0.0,
                )
            )
        assert t.attempts == 3 and t.improved == 2 and t.regressed == 1
        assert t.regression_rate == pytest.approx(1 / 3)

    def test_accept_and_abstain_are_not_correction_attempts(self):
        t = RegressionTracker()
        for action in (Action.ACCEPT, Action.ABSTAIN):
            t.record(
                CorrectionExecutor(provider=_Provider())._outcome(
                    "t", action, VerdictProfile(1, 0, 0), None, True, False, 0.0
                )
            )
        assert t.attempts == 0

    def test_summary_reports_both_directions(self):
        """A stage that improves 60% and degrades 20% is a different system from
        one that improves 45% and degrades none."""
        t = RegressionTracker(attempts=10, improved=6, regressed=2, unchanged=2)
        s = t.summary()
        assert "improved" in s and "regressed" in s

    def test_empty_tracker_summary(self):
        assert RegressionTracker().summary() == "no corrections attempted"

"""Tests for the shared data contracts, focusing on the confidence signals.

These properties are what the fusion model is fitted on. A silent error here does
not crash anything -- it produces a confidence score that looks plausible and is
wrong, which is precisely the failure this project exists to detect.
"""

from __future__ import annotations

import pytest

from pramana.generation.drafting import DraftGenerator, DraftingPolicy
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


def chunk(cid: str = "c1", text: str = "Some policy text.") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", text=text, language="en")


def retrieval(*scores: float) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        normalized_query="q",
        language="en",
        script="roman",
        chunks=[
            RetrievedChunk(chunk=chunk(f"c{i}"), rank=i, fused_score=s)
            for i, s in enumerate(scores)
        ],
    )


def verdict(v: Verdict, *, entail: float = 0.0, contra: float = 0.0, neutral: float = 0.0):
    return ClaimVerdict(
        claim=Claim(claim_id="x", text="a claim", language="en"),
        verdict=v,
        entailment_prob=entail,
        contradiction_prob=contra,
        neutral_prob=neutral,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Signal S1 — retrieval quality
# ──────────────────────────────────────────────────────────────────────────────


class TestRetrievalSignals:
    def test_empty_retrieval_is_detected(self):
        """The generator must abstain rather than answer from nothing -- this is
        the fixture F04 failure mode."""
        assert retrieval().is_empty
        assert not retrieval(0.5).is_empty

    def test_score_margin_discriminates_flat_from_peaked(self):
        """A flat score distribution means retrieval could not tell the passages
        apart, which predicts an ungrounded answer better than the top score does."""
        peaked = retrieval(0.9, 0.2, 0.1)
        flat = retrieval(0.5, 0.49, 0.48)
        assert peaked.score_margin > flat.score_margin
        assert peaked.score_margin == pytest.approx(0.7)

    def test_single_chunk_has_zero_margin(self):
        assert retrieval(0.9).score_margin == 0.0

    def test_empty_retrieval_signals_are_zero_not_undefined(self):
        s = retrieval().signals()
        assert s["s1_top_score"] == 0.0 and s["s1_n_chunks"] == 0.0

    def test_rerank_score_overrides_fused_score(self):
        rc = RetrievedChunk(chunk=chunk(), rank=0, fused_score=0.3, rerank_score=0.9)
        assert rc.score == 0.9

    def test_falls_back_to_fused_when_no_reranker(self):
        assert RetrievedChunk(chunk=chunk(), rank=0, fused_score=0.3).score == 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Signal S4 — self-consistency
# ──────────────────────────────────────────────────────────────────────────────


class TestSelfConsistency:
    def test_identical_samples_score_one(self):
        d = Draft(text="the deadline is thirty days", language="en", samples=["the deadline is thirty days"] * 3)
        assert d.self_consistency == pytest.approx(1.0)
        assert d.sample_dispersion == pytest.approx(0.0)

    def test_divergent_samples_score_low(self):
        d = Draft(
            text="the deadline is thirty days",
            language="en",
            samples=["completely unrelated wording here", "another totally different phrasing"],
        )
        assert d.self_consistency < 0.4

    def test_no_samples_scores_zero_not_one(self):
        """With nothing to compare against, the honest answer is 'no evidence of
        consistency'. Returning 1.0 would tell the fusion model the answer was
        maximally stable when it was never tested."""
        assert Draft(text="anything", language="en").self_consistency == 0.0

    def test_degenerate_samples_are_flagged(self):
        d = Draft(text="same", language="en", samples=["same", "same"])
        assert d.degenerate_samples()

    def test_short_answers_are_marked_unreliable_for_s4(self):
        """Observed live: a Tamil yes/no answer produced three identical samples at
        temperature 0.9. A one-token answer is stable at any temperature, so S4
        is pinned at 1.0 and says nothing about correctness."""
        short = Draft(text="No", language="en", samples=["No", "No", "No"])
        assert short.self_consistency == pytest.approx(1.0)
        assert not short.s4_reliable, "S4 must not be trusted on a one-token answer"

    def test_long_answers_are_reliable_for_s4(self):
        long = Draft(
            text="Your claim was rejected due to late submission after thirty days",
            language="en",
            samples=["Your claim was rejected because it was late", "Rejected for late filing"],
        )
        assert long.s4_reliable

    def test_reliability_flag_is_exposed_as_a_feature(self):
        """The fusion model must be able to learn to discount S4, rather than
        treating a trivially-stable 1.0 as evidence of correctness."""
        s = Draft(text="No", language="en", samples=["No", "No"]).signals()
        assert s["s4_reliable"] == 0.0
        assert s["answer_tokens"] == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Signal S3 — logprobs, and its absence
# ──────────────────────────────────────────────────────────────────────────────


class TestLogprobSignal:
    def test_absent_logprobs_omit_the_feature_entirely(self):
        """S3 must be *missing*, not zero. A zero would be a real, wrong value and
        would corrupt calibration silently; a missing key lets the fusion model be
        refitted without it."""
        s = Draft(text="hello world", language="en").signals()
        assert "s3_mean_logprob" not in s and "s3_perplexity" not in s

    def test_present_logprobs_produce_both_features(self):
        s = Draft(text="hello world", language="en", mean_token_logprob=-0.5).signals()
        assert s["s3_mean_logprob"] == pytest.approx(-0.5)
        assert s["s3_perplexity"] == pytest.approx(1.6487, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# Signal S2 — entailment
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectionSignals:
    def test_counts_each_verdict_class(self):
        d = DetectionResult([
            verdict(Verdict.SUPPORTED),
            verdict(Verdict.CONTRADICTED),
            verdict(Verdict.UNVERIFIABLE),
            verdict(Verdict.SUPPORTED),
        ])
        assert d.count(Verdict.SUPPORTED) == 2
        assert d.supported_ratio == pytest.approx(0.5)
        assert d.hallucination_rate == pytest.approx(0.5)
        assert d.has_contradiction

    def test_margin_measures_verifier_indecision(self):
        """A near-tie matters more than which label won: it means the verifier had
        no real basis for its call."""
        confident = verdict(Verdict.SUPPORTED, entail=0.95, contra=0.02, neutral=0.03)
        uncertain = verdict(Verdict.SUPPORTED, entail=0.40, contra=0.35, neutral=0.25)
        assert confident.margin > uncertain.margin
        assert uncertain.margin == pytest.approx(0.05)

    def test_contradicted_and_unverifiable_both_count_as_hallucination(self):
        assert verdict(Verdict.CONTRADICTED).is_hallucination
        assert verdict(Verdict.UNVERIFIABLE).is_hallucination
        assert not verdict(Verdict.SUPPORTED).is_hallucination

    def test_empty_detection_does_not_divide_by_zero(self):
        d = DetectionResult([])
        assert d.supported_ratio == 0.0 and d.hallucination_rate == 0.0
        assert d.signals()["s2_n_claims"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Drafting policy
# ──────────────────────────────────────────────────────────────────────────────


class TestDraftingPolicy:
    def test_rejects_zero_temperature_with_sampling(self):
        """The whole point of the dual-temperature design: sampling at temperature
        0 yields identical samples and silently reduces S4 to a constant."""
        with pytest.raises(ValueError, match="sample_temperature must be > 0"):
            DraftingPolicy(n_samples=4, sample_temperature=0.0)

    def test_allows_zero_temperature_when_sampling_is_disabled(self):
        p = DraftingPolicy(n_samples=0, sample_temperature=0.0)
        assert p.n_samples == 0

    def test_answer_temperature_stays_deterministic_by_default(self):
        p = DraftingPolicy()
        assert p.answer_temperature == 0.0
        assert p.sample_temperature > 0.0


class TestPrompting:
    @pytest.mark.parametrize("language", ["en", "hi", "ta"])
    def test_every_language_permits_abstention(self, language):
        """A system that may never say "I don't know" must fabricate when the
        evidence is absent."""
        msgs = DraftGenerator.build_messages("q", "ctx", language)
        assert "INSUFFICIENT_EVIDENCE" in msgs[0].content

    @pytest.mark.parametrize("language", ["en", "hi", "ta"])
    def test_context_and_query_reach_the_prompt(self, language):
        msgs = DraftGenerator.build_messages("MYQUERY", "MYCONTEXT", language)
        assert "MYQUERY" in msgs[1].content and "MYCONTEXT" in msgs[1].content

    def test_abstention_is_recognised_when_wrapped_in_prose(self):
        """Models routinely add punctuation or a preamble. Requiring an exact match
        would send an abstention into claim verification as if it were an answer."""
        assert DraftGenerator.is_abstention("INSUFFICIENT_EVIDENCE")
        assert DraftGenerator.is_abstention("  INSUFFICIENT_EVIDENCE.  ")
        assert DraftGenerator.is_abstention("I must reply: INSUFFICIENT_EVIDENCE")
        assert not DraftGenerator.is_abstention("The deadline is 30 days.")

    def test_abstention_message_exists_for_every_language(self):
        for lang in ("en", "hi", "ta"):
            assert DraftGenerator.abstention_message(lang).strip()


class TestActionVocabulary:
    def test_all_five_policy_actions_exist(self):
        assert {a.value for a in Action} == {
            "ACCEPT",
            "REGENERATE",
            "RE_RETRIEVE",
            "PRUNE",
            "ABSTAIN",
        }

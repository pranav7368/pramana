"""End-to-end tests for the assembled pipeline.

Driven entirely offline through the stub provider and the keyword verifier, so the
full five-stage flow is exercised with no GPU, no network and no API key. Every
assertion is about *behaviour on a known defect*, not about the code merely
running.
"""

from __future__ import annotations

import pytest

from pramana.confidence.fusion import ConfidenceModel
from pramana.correction.policy import CorrectionExecutor, CorrectionPolicy
from pramana.detection.decomposer import RuleBasedDecomposer
from pramana.detection.verifier import GroundingVerifier, KeywordNLIBackend
from pramana.generation.base import Completion, GenerationResponse
from pramana.generation.drafting import DraftGenerator, DraftingPolicy
from pramana.ingestion.chunking import chunk_text
from pramana.pipeline import PramanaPipeline, result_summary
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever
from pramana.schemas import Action, Verdict

CORPUS = {
    "en": (
        "Claims are rejected if submitted more than 30 days after the discharge date. "
        "Rejected claims may be appealed within 60 days of the rejection notice. "
        "Maternity benefits are not available during the first policy year."
    ),
    "hi": (
        "डिस्चार्ज की तारीख से 30 दिन बाद जमा किए गए क्लेम रिजेक्ट कर दिए जाते हैं। "
        "रिजेक्ट क्लेम की अपील 60 दिन के भीतर की जा सकती है।"
    ),
}


class _FixedProvider:
    """Returns a scripted answer per call. Makes the whole pipeline deterministic."""

    name = "fixed"

    def __init__(self, *replies: str):
        self.queue = list(replies)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        text = self.queue.pop(0) if self.queue else (self.queue[-1] if self.queue else "")
        return GenerationResponse(completions=[Completion(text)], model="m", provider=self.name)

    def capabilities(self, model):
        return set()

    def available_models(self):
        return ["m"]

    def health_check(self):
        return True


def build(provider, *, correct: bool = True, languages=("en",)) -> PramanaPipeline:
    retriever = MultilingualRetriever()
    for lang in languages:
        r = HybridRetriever(language=lang, top_k=3)
        r.add(chunk_text(CORPUS[lang], doc_id=f"policy_{lang}", language=lang))
        retriever.add_language(lang, r)

    return PramanaPipeline(
        retriever=retriever,
        generator=DraftGenerator(provider, DraftingPolicy(n_samples=0, sample_temperature=0.7)),
        decomposer=RuleBasedDecomposer(),
        verifier=GroundingVerifier(backend=KeywordNLIBackend()),
        confidence=ConfidenceModel(),
        policy=CorrectionPolicy(max_iterations=2),
        executor=CorrectionExecutor(provider=provider) if correct else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────────


class TestGroundedAnswer:
    def test_faithful_answer_is_accepted(self):
        p = build(_FixedProvider("Rejected claims may be appealed within 60 days of the rejection notice."))
        r = p.run("How long do I have to appeal?")

        assert r.action_history[-1] is Action.ACCEPT
        assert not r.regressed and not r.abstained
        assert r.detection.count(Verdict.SUPPORTED) >= 1

    def test_evidence_is_attributed(self):
        """A citation is only useful if it names the chunk that supports the claim."""
        p = build(_FixedProvider("Rejected claims may be appealed within 60 days of the rejection notice."))
        r = p.run("How long do I have to appeal?")
        assert r.evidence_chunk_ids
        assert all(cid.startswith("policy_en#") for cid in r.evidence_chunk_ids)

    def test_result_carries_a_trace_and_timings(self):
        r = build(_FixedProvider("Rejected claims may be appealed within 60 days.")).run("What is the appeal window for a rejected claim?")
        assert r.trace_id
        assert {"language", "retrieval", "generation"} <= set(r.latency_ms)
        assert r.total_latency_ms >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination handling
# ──────────────────────────────────────────────────────────────────────────────


class TestHallucinationDetection:
    def test_numeric_contradiction_is_detected(self):
        """Fixture F01's defect, driven through the whole pipeline."""
        p = build(_FixedProvider("Claims are rejected if submitted more than 15 days after discharge."), correct=False)
        r = p.run("When are claims rejected after discharge?")
        assert r.detection.has_contradiction

    def test_contradiction_triggers_correction(self):
        p = build(
            _FixedProvider(
                "Claims are rejected if submitted more than 15 days after discharge.",
                "Claims are rejected if submitted more than 30 days after discharge.",
            )
        )
        r = p.run("When are claims rejected after discharge?")
        assert Action.REGENERATE in r.action_history

    def test_successful_correction_replaces_the_answer(self):
        p = build(
            _FixedProvider(
                "Claims are rejected if submitted more than 15 days after discharge.",
                "Claims are rejected if submitted more than 30 days after discharge.",
            )
        )
        r = p.run("When are claims rejected after discharge?")
        assert "30 days" in r.final_answer, "corrected answer should have replaced the draft"
        assert not r.regressed
        assert r.iterations >= 1

    def test_unsupported_claim_is_flagged(self):
        p = build(_FixedProvider("The policy includes free gym membership for all employees."), correct=False)
        r = p.run("What does the policy cover for claims?")
        assert r.detection.hallucination_rate > 0


# ──────────────────────────────────────────────────────────────────────────────
# The guard rails
# ──────────────────────────────────────────────────────────────────────────────


class TestGuardRails:
    def test_failed_correction_is_rolled_back(self):
        """The documented risk: self-correction can degrade a correct answer.
        A revision that does not strictly improve must be discarded, and the fact
        recorded rather than hidden."""
        original = "Claims are rejected if submitted more than 15 days after discharge."
        p = build(_FixedProvider(original, "Maternity benefits are available in the first policy year."))
        r = p.run("When are claims rejected after discharge?")

        assert r.rolled_back and not r.regressed, "equal profiles must be rolled back without inflating regression"
        assert r.final_answer == original, "the original answer must be restored"

    def test_loop_depth_is_bounded(self):
        """Cost must be provable, not emergent."""
        wrong = "Claims are rejected if submitted more than 15 days after discharge."
        p = build(_FixedProvider(*[wrong] * 10))
        p.policy.max_iterations = 2
        r = p.run("When are claims rejected after discharge?")
        assert r.iterations <= 2
        assert len(r.action_history) <= 4

    def test_regression_tracker_counts_attempts(self):
        p = build(
            _FixedProvider(
                "Claims are rejected if submitted more than 15 days after discharge.",
                "Claims are rejected if submitted more than 30 days after discharge.",
            )
        )
        p.run("When are claims rejected after discharge?")
        assert p.tracker.attempts >= 1
        assert p.tracker.improved + p.tracker.regressed + p.tracker.unchanged == p.tracker.attempts

    def test_correction_can_be_disabled_for_ablation(self):
        """The detection-only arm must run without a correction executor."""
        p = build(_FixedProvider("Claims are rejected after 15 days."), correct=False)
        r = p.run("When are claims rejected after discharge?")
        assert r.iterations == 0
        assert not r.regressed


# ──────────────────────────────────────────────────────────────────────────────
# Abstention
# ──────────────────────────────────────────────────────────────────────────────


class TestAbstention:
    def test_abstention_short_circuits_before_verification(self):
        """Decomposing an abstention yields a claim scored UNVERIFIABLE, counting
        the system as hallucinating for correctly declining. It must never
        reach the verifier."""
        r = build(_FixedProvider("INSUFFICIENT_EVIDENCE")).run("What is the WiFi password?")

        assert r.abstained
        assert r.detection.n_claims == 0, "an abstention must not be decomposed"
        assert r.detection.hallucination_rate == 0.0, "declining is not a hallucination"

    def test_abstention_returns_a_readable_message(self):
        r = build(_FixedProvider("INSUFFICIENT_EVIDENCE")).run("What is the appeal window for a rejected claim?")
        assert "INSUFFICIENT_EVIDENCE" not in r.final_answer
        assert r.final_answer.strip()

    def test_empty_retrieval_abstains_without_calling_the_model(self):
        """No evidence means the answer is foregone; spending a generation call to
        reach it wastes quota and invites a fluent ungrounded answer."""
        provider = _FixedProvider("some answer")
        p = build(provider)
        p.retriever.retrievers["en"] = HybridRetriever(language="en")  # empty index

        r = p.run("anything at all")
        assert r.abstained
        assert provider.calls == 0, "must not generate against empty evidence"


# ──────────────────────────────────────────────────────────────────────────────
# Multilingual routing
# ──────────────────────────────────────────────────────────────────────────────


class TestMultilingual:
    def test_hindi_query_routes_to_the_hindi_corpus(self):
        p = build(
            _FixedProvider("रिजेक्ट क्लेम की अपील 60 दिन के भीतर की जा सकती है।"),
            languages=("en", "hi"),
        )
        r = p.run("अपील कितने दिन में कर सकते हैं?")
        assert r.language == "hi"
        assert all(cid.startswith("policy_hi#") for cid in r.evidence_chunk_ids)

    def test_explicit_language_overrides_detection(self):
        p = build(_FixedProvider("some answer here"), languages=("en", "hi"))
        assert p.run("ambiguous", language="hi").language == "hi"

    def test_hindi_contradiction_is_detected(self):
        p = build(
            _FixedProvider("डिस्चार्ज की तारीख से 15 दिन बाद जमा किए गए क्लेम रिजेक्ट कर दिए जाते हैं।"),
            correct=False,
            languages=("en", "hi"),
        )
        r = p.run("क्लेम जमा करने की समय सीमा क्या है?")
        assert r.detection.has_contradiction


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────


class TestReporting:
    def test_summary_includes_the_decision_trail(self):
        r = build(_FixedProvider("Rejected claims may be appealed within 60 days.")).run("What is the appeal window for a rejected claim?")
        s = result_summary(r)
        assert "conf=" in s and "actions=" in s and "claims=" in s

    def test_summary_flags_abstention(self):
        r = build(_FixedProvider("INSUFFICIENT_EVIDENCE")).run("What is the appeal window for a rejected claim?")
        assert "ABSTAINED" in result_summary(r)

    def test_summary_flags_regression(self):
        original = "Claims are rejected if submitted more than 15 days after discharge."
        r = build(_FixedProvider(original, "Maternity benefits are available in the first policy year.")).run("What is the appeal window for a rejected claim?")
        assert "ROLLED_BACK" in result_summary(r)

    def test_untrained_confidence_is_marked(self):
        """An untrained heuristic must never be mistaken for a calibrated score."""
        r = build(_FixedProvider("Rejected claims may be appealed within 60 days.")).run("What is the appeal window for a rejected claim?")
        assert r.confidence.calibrator in ("none", "n/a")

    def test_result_serialises(self):
        r = build(_FixedProvider("Rejected claims may be appealed within 60 days.")).run("What is the appeal window for a rejected claim?")
        d = r.to_dict()
        assert d["trace_id"] == r.trace_id
        assert "confidence" in d and "detection" in d


@pytest.mark.parametrize("query", ["", "   ", "???", "a"])
def test_degenerate_queries_do_not_crash(query):
    """Live traffic contains empty and junk queries; they must degrade, not raise."""
    r = build(_FixedProvider("INSUFFICIENT_EVIDENCE")).run(query)
    assert r.trace_id and r.final_answer.strip()

"""Tests for claim decomposition and NLI grounding verification.

The pipeline is driven end-to-end against the planted-hallucination fixtures, so
these assert **correctness** -- the right verdict for the right reason -- rather
than merely that nothing crashed.
"""

from __future__ import annotations

import pytest

from pramana.detection.decomposer import (
    LLMDecomposer,
    RuleBasedDecomposer,
    decomposition_report,
)
from pramana.detection.verifier import (
    GroundingVerifier,
    KeywordNLIBackend,
    LLMNLIBackend,
    NLIScores,
    Thresholds,
    apply_thresholds,
    detection_report,
)
from pramana.generation.base import Completion, GenerationResponse
from pramana.generation.stub import FIXTURES_BY_ID
from pramana.schemas import Chunk, Claim, Draft, RetrievalResult, RetrievedChunk, Verdict


def draft(text: str, language="en") -> Draft:
    return Draft(text=text, language=language)


def retrieval(*texts: str, language="en") -> RetrievalResult:
    return RetrievalResult(
        query="q",
        normalized_query="q",
        language=language,
        script="native" if language != "en" else "roman",
        chunks=[
            RetrievedChunk(
                chunk=Chunk(f"c{i}", "d1", t, language), rank=i, fused_score=1.0 - i * 0.1
            )
            for i, t in enumerate(texts)
        ],
    )


class _ScriptedProvider:
    """Returns queued responses in order. Lets prompt-driven components be tested
    deterministically without a network call."""

    name = "scripted"

    def __init__(self, *responses: str):
        self.queue = list(responses)
        self.prompts: list[str] = []

    def generate(self, request):
        self.prompts.append("\n".join(m.content for m in request.messages))
        text = self.queue.pop(0) if self.queue else ""
        return GenerationResponse(
            completions=[Completion(text=text)], model="m", provider=self.name
        )

    def capabilities(self, model):
        return set()

    def available_models(self):
        return ["m"]

    def health_check(self):
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Decomposition
# ──────────────────────────────────────────────────────────────────────────────


class TestRuleBasedDecomposer:
    def test_splits_sentences_into_claims(self):
        d = RuleBasedDecomposer()
        claims = d.decompose(draft("The claim was rejected. An appeal is allowed within 60 days."))
        assert len(claims) >= 2

    def test_splits_on_causal_connective(self):
        """'X because Y' is two independently verifiable propositions."""
        claims = RuleBasedDecomposer().decompose(
            draft("The claim was rejected because it was submitted late.")
        )
        assert len(claims) == 2

    def test_claims_are_span_linked(self):
        text = "The claim was rejected. An appeal is allowed within 60 days."
        for c in RuleBasedDecomposer().decompose(draft(text)):
            assert c.source_span is not None
            start, end = c.source_span
            assert text[start:end] == c.text

    def test_drops_fragments_with_no_content(self):
        """A bare connective sent to the verifier yields an UNVERIFIABLE verdict
        that is real in form and meaningless in substance, inflating the measured
        hallucination rate."""
        claims = RuleBasedDecomposer().decompose(draft("Yes. The deadline is 30 days after discharge."))
        assert all(len(c.text) >= 8 for c in claims)

    def test_empty_draft_yields_no_claims(self):
        assert RuleBasedDecomposer().decompose(draft("")) == []

    def test_claim_ids_are_unique(self):
        claims = RuleBasedDecomposer().decompose(
            draft("First fact here. Second fact here. Third fact here.")
        )
        assert len({c.claim_id for c in claims}) == len(claims)

    def test_handles_hindi_danda(self):
        claims = RuleBasedDecomposer().decompose(
            draft("क्लेम रिजेक्ट हो गया। अपील 60 दिन में की जा सकती है।", "hi")
        )
        assert len(claims) == 2


class TestLLMDecomposer:
    def test_parses_one_claim_per_line(self):
        p = _ScriptedProvider("The claim was rejected.\nThe deadline is 30 days.")
        claims = LLMDecomposer(p).decompose(draft("Your claim was rejected; the deadline is 30 days."))
        assert [c.text for c in claims] == ["The claim was rejected.", "The deadline is 30 days."]

    def test_strips_list_markers(self):
        p = _ScriptedProvider("1. The claim was rejected.\n- The deadline is 30 days.")
        claims = LLMDecomposer(p).decompose(
            draft("Your claim was rejected and the submission deadline is 30 days.")
        )
        assert claims[0].text == "The claim was rejected."
        assert claims[1].text == "The deadline is 30 days."

    def test_no_claims_marker_yields_empty(self):
        assert LLMDecomposer(_ScriptedProvider("NO_CLAIMS")).decompose(draft("Hello there!")) == []

    def test_falls_back_when_the_model_fails(self):
        """A decomposition failure must degrade to coarser claims, not abort
        verification of the whole answer."""

        class _Broken(_ScriptedProvider):
            def generate(self, request):
                raise RuntimeError("provider exploded")

        claims = LLMDecomposer(_Broken()).decompose(
            draft("The claim was rejected. An appeal is allowed.")
        )
        assert claims, "should have fallen back to rule-based decomposition"

    def test_falls_back_on_unusable_output(self):
        claims = LLMDecomposer(_ScriptedProvider("   \n  \n ")).decompose(
            draft("The claim was rejected. An appeal is allowed.")
        )
        assert claims

    def test_preserves_answer_when_decomposition_cap_is_exceeded(self):
        """A model that emits a list instead of claims must not produce hundreds
        of verification calls."""
        long_answer = " ".join(f"Fact number {i} is stated here clearly." for i in range(80))
        p = _ScriptedProvider("\n".join(f"Claim number {i} is stated." for i in range(50)))
        claims = LLMDecomposer(p, max_claims=20).decompose(draft(long_answer))
        assert len(claims) == 80
        assert "79" in claims[-1].text, "the answer tail must not escape verification"

    def test_rewritten_claims_have_no_span(self):
        """A Hindi claim rewritten to restore an elided subject no longer occurs
        verbatim in the draft. Reporting a span anyway would make PRUNE excise
        the wrong text."""
        p = _ScriptedProvider("क्लेम रिजेक्ट हो गया।")
        claims = LLMDecomposer(p).decompose(draft("रिजेक्ट हो गया।", "hi"))
        assert claims[0].source_span is None

    def test_abstention_never_enters_verification(self):
        """Decomposing "INSUFFICIENT_EVIDENCE" yields the claim "There is
        insufficient evidence", which is then scored UNVERIFIABLE and counted as a
        hallucination -- penalising the system for correctly declining to answer.
        Observed live 2026-08-10."""
        p = _ScriptedProvider("There is insufficient evidence.")
        assert LLMDecomposer(p).decompose(draft("INSUFFICIENT_EVIDENCE")) == []
        assert RuleBasedDecomposer().decompose(draft("INSUFFICIENT_EVIDENCE")) == []

    def test_very_short_answers_are_not_decomposed(self):
        """A two-word answer has nothing to decompose, and asking anyway invites
        invented claims."""
        p = _ScriptedProvider("Thirty days is a specific deadline.")
        assert LLMDecomposer(p).decompose(draft("30 days.")) == []

    def test_inflated_decomposition_falls_back(self):
        """Observed live: given the terse Tamil answer "30 நாட்கள், ஆம்."
        ("30 days, yes.") the model returned "30 days is a specific deadline" and
        "Yes indicates agreement" -- meta-commentary about the answer rather than
        propositions in it. The second was scored UNVERIFIABLE, reporting a 50%
        hallucination rate for a correct answer. A decomposition artefact must
        never be counted as a generator failure."""
        source = "The deadline is 30 days and appeals are allowed."
        invented = "\n".join(
            f"This statement number {i} explains an additional invented concept in detail."
            for i in range(8)
        )
        claims = LLMDecomposer(_ScriptedProvider(invented)).decompose(draft(source))
        assert all(c.text in source or len(c.text) < len(source) for c in claims)
        assert sum(len(c.text) for c in claims) <= max(60, len(source) * 2)

    def test_legitimate_expansion_is_allowed(self):
        """Restoring an elided subject genuinely lengthens a claim, so the
        inflation guard must not fire on correct pro-drop handling."""
        p = _ScriptedProvider("क्लेम रिजेक्ट हो गया।\nक्लेम 30 दिन के अंदर जमा नहीं किया गया।")
        claims = LLMDecomposer(p).decompose(
            draft("रिजेक्ट हो गया क्योंकि 30 दिन के अंदर जमा नहीं किया गया।", "hi")
        )
        assert len(claims) == 2
        assert all("क्लेम" in c.text for c in claims), "subject must be restored"

    def test_prompt_forbids_meta_commentary(self):
        p = _ScriptedProvider("A claim about the deadline.")
        LLMDecomposer(p).decompose(draft("The deadline is 30 days after discharge."))
        assert "commentary about the answer" in p.prompts[0]

    def test_prompt_includes_a_subject_restoration_example(self):
        """Pro-drop handling is the behaviour most likely to be dropped, so the
        few-shot example must demonstrate it."""
        p = _ScriptedProvider("x claim here")
        LLMDecomposer(p).decompose(draft("रिजेक्ट हो गया।", "hi"))
        assert "क्लेम रिजेक्ट हो गया" in p.prompts[0]


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────


class TestThresholds:
    def test_contradiction_takes_precedence(self):
        """An answer that conflicts with the evidence is a worse failure than one
        the evidence merely fails to confirm, and needs a different correction."""
        scores = NLIScores(entailment=0.60, contradiction=0.60, neutral=0.0)
        assert apply_thresholds(scores, Thresholds(entail=0.55, contra=0.55)) is Verdict.CONTRADICTED

    def test_below_both_thresholds_is_unverifiable(self):
        scores = NLIScores(entailment=0.40, contradiction=0.30, neutral=0.30)
        assert apply_thresholds(scores, Thresholds(0.55, 0.50)) is Verdict.UNVERIFIABLE

    def test_thresholds_differ_by_language(self):
        """Multilingual NLI models are not equally calibrated: entailment
        probabilities run lower for Tamil, so one global threshold measurably
        penalises it."""
        d = Thresholds.defaults()
        assert d["ta"].entail < d["en"].entail

    def test_a_borderline_claim_flips_with_the_language_threshold(self):
        scores = NLIScores(entailment=0.52, contradiction=0.10, neutral=0.38)
        d = Thresholds.defaults()
        assert apply_thresholds(scores, d["en"]) is Verdict.UNVERIFIABLE
        assert apply_thresholds(scores, d["ta"]) is Verdict.SUPPORTED


class TestNLIScores:
    def test_normalisation_sums_to_one(self):
        s = NLIScores(2.0, 1.0, 1.0).normalised()
        assert s.entailment + s.contradiction + s.neutral == pytest.approx(1.0)

    def test_degenerate_scores_become_neutral(self):
        """All-zero scores must not normalise into a confident verdict."""
        assert NLIScores(0.0, 0.0, 0.0).normalised().label == "neutral"


# ──────────────────────────────────────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────────────────────────────────────


class TestKeywordBackend:
    def test_detects_numeric_contradiction(self):
        """Fixture F01: 30 days in the evidence, 15 in the claim."""
        s = KeywordNLIBackend().score(
            "Claims are rejected if submitted more than 30 days after discharge.",
            "The submission deadline is 15 days after discharge.",
            "en",
        )
        assert s.label == "contradiction"

    def test_detects_dropped_negation(self):
        """Fixture F05, and the error class machine translation introduces most."""
        s = KeywordNLIBackend().score(
            "Maternity benefits are not available during the first policy year.",
            "Maternity benefits are available during the first policy year.",
            "en",
        )
        assert s.label == "contradiction"

    def test_supports_a_grounded_claim(self):
        s = KeywordNLIBackend().score(
            "Rejected claims may be appealed within 60 days of the rejection notice.",
            "An appeal may be filed within 60 days of the rejection notice.",
            "en",
        )
        assert s.label == "entailment"

    def test_unrelated_claim_is_neutral(self):
        s = KeywordNLIBackend().score(
            "Claims are rejected if submitted late.",
            "The office cafeteria serves lunch at noon.",
            "en",
        )
        assert s.label == "neutral"

    def test_number_from_an_unrelated_sentence_does_not_mask_a_contradiction(self):
        """Observed in the demo: the claim "not submitted within 15 days" was
        marked SUPPORTED against a chunk stating 30 days, because the same chunk
        mentioned "15 unused leave days" in an unrelated sentence. Surface
        heuristics must be scoped to the sentence they apply to."""
        s = KeywordNLIBackend().score(
            "Claims are rejected if submitted more than 30 days after discharge. "
            "Employees may carry forward up to 15 unused leave days into the next year.",
            "The claim was not submitted within 15 days of discharge.",
            "en",
        )
        assert s.label == "contradiction", "a borrowed number must not mask the conflict"

    def test_negation_from_an_unrelated_sentence_is_ignored(self):
        """The mirror case: a "not" elsewhere in the chunk must not invent a
        contradiction. This one rejected correct corrections and inflated the
        regression rate in controlled fixtures."""
        s = KeywordNLIBackend().score(
            "Claims are rejected if submitted more than 30 days after the discharge date. "
            "Maternity benefits are not available during the first policy year.",
            "Claims are rejected if submitted more than 30 days after discharge.",
            "en",
        )
        assert s.label != "contradiction"

    def test_negation_flip_needs_topical_overlap(self):
        """Otherwise every unrelated sentence containing 'not' reads as a
        contradiction."""
        s = KeywordNLIBackend().score(
            "The cafeteria is not open on Sundays.",
            "Claims must be submitted within 30 days.",
            "en",
        )
        assert s.label != "contradiction"


class TestLLMBackend:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("ENTAILMENT", "entailment"),
            ("CONTRADICTION", "contradiction"),
            ("NEUTRAL", "neutral"),
            ("The answer is CONTRADICTION.", "contradiction"),
            ("entailment", "entailment"),
        ],
    )
    def test_parses_labels_including_wrapped_ones(self, reply, expected):
        b = LLMNLIBackend(_ScriptedProvider(reply))
        assert b.score("p", "h", "en").label == expected

    def test_unparseable_output_becomes_neutral_not_supported(self):
        """An unreadable judgement must never become an accidental SUPPORTED."""
        b = LLMNLIBackend(_ScriptedProvider("I'm not sure about this one"))
        assert b.score("p", "h", "en").label == "neutral"

    def test_backend_failure_becomes_neutral(self):
        class _Broken(_ScriptedProvider):
            def generate(self, request):
                raise RuntimeError("down")

        assert LLMNLIBackend(_Broken()).score("p", "h", "en").label == "neutral"

    def test_prompt_warns_about_numbers_and_negation(self):
        p = _ScriptedProvider("NEUTRAL")
        LLMNLIBackend(p).score("premise", "hypothesis", "en")
        assert "negation" in p.prompts[0].lower()


# ──────────────────────────────────────────────────────────────────────────────
# Verifier
# ──────────────────────────────────────────────────────────────────────────────


class TestGroundingVerifier:
    def verifier(self) -> GroundingVerifier:
        return GroundingVerifier(backend=KeywordNLIBackend())

    def claims(self, *texts, language="en") -> list[Claim]:
        return [Claim(f"c{i}", t, language) for i, t in enumerate(texts)]

    def test_empty_retrieval_makes_every_claim_unverifiable(self):
        """Fixture F04's path -- and it must not spend a backend call to reach a
        foregone conclusion."""
        result = self.verifier().verify(
            self.claims("Dental cover is Rs. 25,000 per year."), retrieval()
        )
        assert result.n_claims == 1
        assert result.claim_verdicts[0].verdict is Verdict.UNVERIFIABLE
        assert result.hallucination_rate == 1.0

    def test_no_claims_yields_empty_result(self):
        assert self.verifier().verify([], retrieval("some evidence")).n_claims == 0

    def test_supported_claim_gets_evidence_attribution(self):
        """A citation is only trustworthy if it points at the chunk that actually
        supports the claim."""
        result = self.verifier().verify(
            self.claims("An appeal may be filed within 60 days of the rejection notice."),
            retrieval(
                "The cafeteria serves lunch at noon.",
                "Rejected claims may be appealed within 60 days of the rejection notice.",
            ),
        )
        v = result.claim_verdicts[0]
        assert v.verdict is Verdict.SUPPORTED
        assert v.supporting_chunk_ids == ["c1"], "must cite the chunk that supports it"

    def test_detects_contradiction_across_chunks(self):
        result = self.verifier().verify(
            self.claims("The submission deadline is 15 days after discharge."),
            retrieval("Claims are rejected if submitted more than 30 days after discharge."),
        )
        assert result.claim_verdicts[0].verdict is Verdict.CONTRADICTED
        assert result.has_contradiction

    def test_support_from_one_chunk_outweighs_contradiction_from_another(self):
        """Enterprise corpora carry general rules alongside their exceptions, so a
        single disagreeing passage usually means the claim is grounded in a
        specific provision. Documented as a design choice and an ablation
        candidate -- the opposite rule is defensible."""
        result = self.verifier().verify(
            self.claims("An appeal may be filed within 60 days of the rejection notice."),
            retrieval(
                "Appeals are not permitted after 15 days.",
                "Rejected claims may be appealed within 60 days of the rejection notice.",
            ),
        )
        assert result.claim_verdicts[0].verdict is Verdict.SUPPORTED

    def test_mixed_verdicts_are_counted_correctly(self):
        result = self.verifier().verify(
            self.claims(
                "An appeal may be filed within 60 days of the rejection notice.",
                "The submission deadline is 15 days after discharge.",
                "The policy includes free gym membership for all employees.",
            ),
            retrieval(
                "Rejected claims may be appealed within 60 days of the rejection notice.",
                "Claims are rejected if submitted more than 30 days after discharge.",
            ),
        )
        assert result.count(Verdict.SUPPORTED) == 1
        assert result.count(Verdict.CONTRADICTED) == 1
        assert result.count(Verdict.UNVERIFIABLE) == 1
        assert result.hallucination_rate == pytest.approx(2 / 3)

    def test_s2_signals_are_populated(self):
        result = self.verifier().verify(
            self.claims("An appeal may be filed within 60 days of the rejection notice."),
            retrieval("Rejected claims may be appealed within 60 days of the rejection notice."),
        )
        s = result.signals()
        assert s["s2_supported_ratio"] == 1.0
        assert s["s2_n_claims"] == 1.0
        assert 0.0 <= s["s2_mean_margin"] <= 1.0

    def test_chunk_budget_is_respected(self):
        """Verification cost is claims x chunks and is the pipeline's dominant
        expense, so the cap must actually bound backend calls."""

        class _Counting:
            name = "counting"

            def __init__(self):
                self.inner = KeywordNLIBackend()
                self.calls = 0

            def score(self, premise, hypothesis, language):
                self.calls += 1
                return self.inner.score(premise, hypothesis, language)

        backend = _Counting()
        v = GroundingVerifier(backend=backend, max_chunks_per_claim=2)
        v.verify(
            self.claims("a claim about something"),
            retrieval(*[f"chunk {i} text" for i in range(9)]),
        )
        assert backend.calls == 2, "cap must bound calls, not just the result list"

    def test_cost_scales_as_claims_times_chunks(self):
        class _Counting:
            name = "counting"

            def __init__(self):
                self.inner = KeywordNLIBackend()
                self.calls = 0

            def score(self, premise, hypothesis, language):
                self.calls += 1
                return self.inner.score(premise, hypothesis, language)

        backend = _Counting()
        GroundingVerifier(backend=backend, max_chunks_per_claim=3).verify(
            self.claims("claim one text", "claim two text", "claim three text", "claim four text"),
            retrieval("chunk a text", "chunk b text", "chunk c text"),
        )
        assert backend.calls == 4 * 3

    def test_records_the_verification_arm(self):
        v = GroundingVerifier(backend=KeywordNLIBackend(), arm="translate")
        result = v.verify(self.claims("some claim text here"), retrieval("some evidence"))
        assert result.claim_verdicts[0].arm == "translate"


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end against the planted fixtures
# ──────────────────────────────────────────────────────────────────────────────


class TestFixtureRecovery:
    """Decompose + verify each fixture and check the planted defect is recovered.

    The keyword backend is a deliberately weak floor: fixtures it cannot resolve
    are the ones that justify a real NLI model, and are marked as such.
    """

    def run(self, fixture_id: str):
        f = FIXTURES_BY_ID[fixture_id]
        d = Draft(text=f.answer, language=f.language)
        claims = RuleBasedDecomposer().decompose(d)
        rr = retrieval(*f.context, language=f.language) if f.context else retrieval(language=f.language)
        return f, GroundingVerifier(backend=KeywordNLIBackend()).verify(claims, rr, f.language)

    def test_f01_numeric_contradiction_is_caught(self):
        f, result = self.run("F01")
        assert result.has_contradiction, f"planted defect missed: {f.defect}"

    def test_f03_fully_grounded_answer_is_not_flagged(self):
        """The negative control. A detector that flags everything scores perfect
        recall and is useless."""
        _, result = self.run("F03")
        assert not result.has_contradiction
        assert result.count(Verdict.SUPPORTED) >= 1

    def test_f04_empty_retrieval_is_all_unverifiable(self):
        _, result = self.run("F04")
        assert result.n_claims > 0
        assert result.count(Verdict.UNVERIFIABLE) == result.n_claims

    def test_f05_dropped_negation_is_caught(self):
        f, result = self.run("F05")
        assert result.has_contradiction, f"planted defect missed: {f.defect}"

    def test_f06_hindi_contradiction_is_caught(self):
        f, result = self.run("F06")
        assert result.has_contradiction, f"planted defect missed: {f.defect}"

    def test_f07_tamil_contradiction_is_caught(self):
        f, result = self.run("F07")
        assert result.has_contradiction, f"planted defect missed: {f.defect}"


class TestReports:
    def test_detection_report_summarises(self):
        v = GroundingVerifier(backend=KeywordNLIBackend())
        result = v.verify(
            [Claim("c0", "An appeal may be filed within 60 days of the notice.", "en")],
            retrieval("Rejected claims may be appealed within 60 days of the notice."),
        )
        out = detection_report(result)
        assert "supported" in out and "hallucination rate" in out

    def test_empty_reports(self):
        assert detection_report(GroundingVerifier(backend=KeywordNLIBackend()).verify([], retrieval())) == "0 claims"
        assert decomposition_report([]) == "0 claims"

    def test_decomposition_report_counts_span_linking(self):
        claims = RuleBasedDecomposer().decompose(draft("The claim was rejected. An appeal is allowed."))
        assert "span-linked" in decomposition_report(claims)

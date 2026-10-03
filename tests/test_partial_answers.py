"""Partly answerable questions: release what the evidence supports, say what it does not.

Asked "How many leave days can I carry forward, and can I encash the rest?"
against a policy that only states the carry-forward rule, the fail-closed
pipeline used to withhold the supported half as well: the whole-answer check
could not entail an answer that leaves part of the question open. A second
judgement now separates an honest partial answer from one that dropped an
unsupported clause or does not address the question.
"""
from __future__ import annotations

import pytest

from pramana.confidence.fusion import ConfidenceModel
from pramana.correction.policy import CorrectionExecutor, CorrectionPolicy
from pramana.detection.decomposer import RuleBasedDecomposer
from pramana.detection.verifier import (
    GroundingVerifier,
    KeywordNLIBackend,
    LLMNLIBackend,
    NLIScores,
)
from pramana.generation.base import Completion, GenerationResponse
from pramana.generation.drafting import PARTIAL_NOTES, DraftGenerator, DraftingPolicy
from pramana.ingestion.chunking import chunk_text
from pramana.pipeline import PramanaPipeline
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever

POLICY = "Employees may carry forward up to 15 unused leave days into the next year."
QUESTION = "How many leave days can I carry forward, and can I encash the rest?"
NEUTRAL = NLIScores(0.05, 0.05, 0.9)


class _Provider:
    name = "fixed"

    def __init__(self, *replies: str):
        self.queue = list(replies)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        text = self.queue.pop(0) if self.queue else ""
        return GenerationResponse(completions=[Completion(text)], model="m", provider=self.name)

    def capabilities(self, model):
        return set()


class _Judge:
    """Keyword NLI for claims; scripted whole-answer and coverage judgements."""

    def __init__(self, coverage: str, whole_answer: NLIScores = NEUTRAL):
        self.inner = KeywordNLIBackend()
        self.coverage = coverage
        self.whole_answer = whole_answer
        self.coverage_calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def score_answer(self, premise, question, answer, language):
        return self.whole_answer

    def assess_coverage(self, premise, question, answer, language):
        self.coverage_calls += 1
        return self.coverage


def _pipeline(judge, answer: str = POLICY) -> PramanaPipeline:
    retriever = MultilingualRetriever()
    index = HybridRetriever(language="en", top_k=3)
    index.add(chunk_text(POLICY, doc_id="policy", language="en"))
    retriever.add_language("en", index)
    provider = _Provider(answer)
    return PramanaPipeline(
        retriever=retriever, generator=DraftGenerator(provider, DraftingPolicy(n_samples=0)),
        decomposer=RuleBasedDecomposer(), verifier=GroundingVerifier(backend=judge),
        confidence=ConfidenceModel(), policy=CorrectionPolicy(max_iterations=0),
        executor=CorrectionExecutor(provider=provider), fail_closed=True,
    )


class TestPipeline:
    def test_supported_partial_answer_is_released_with_a_notice(self):
        judge = _Judge("PARTIAL")
        result = _pipeline(judge).run(QUESTION, language="en")
        assert not result.abstained and result.partial and result.stop_reason == "accept"
        assert result.final_answer.startswith(POLICY)
        assert result.final_answer.endswith(PARTIAL_NOTES["en"])
        assert result.detection.hallucination_rate == 0 and judge.coverage_calls == 1

    @pytest.mark.parametrize("label", ["UNRELATED", "UNSUPPORTED", "FULL"])
    def test_every_other_coverage_judgement_still_withholds(self, label):
        result = _pipeline(_Judge(label)).run(QUESTION, language="en")
        assert result.abstained and not result.partial
        assert result.final_answer == DraftGenerator.abstention_message("en")

    def test_coverage_is_not_asked_when_the_whole_answer_is_entailed(self):
        judge = _Judge("PARTIAL", whole_answer=NLIScores(0.9, 0.05, 0.05))
        result = _pipeline(judge).run("How many leave days can I carry forward?", language="en")
        assert not result.abstained and not result.partial and judge.coverage_calls == 0

    def test_a_contradicted_whole_answer_never_becomes_partial(self):
        judge = _Judge("PARTIAL", whole_answer=NLIScores(0.05, 0.9, 0.05))
        result = _pipeline(judge).run(QUESTION, language="en")
        assert result.abstained and judge.coverage_calls == 0

    def test_backends_without_a_coverage_check_keep_the_old_behaviour(self):
        class Plain:
            """A judge with a whole-answer check but no coverage check."""

            def __init__(self):
                self.inner = KeywordNLIBackend()

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def score_answer(self, premise, question, answer, language):
                return NEUTRAL

        assert _pipeline(Plain()).run(QUESTION, language="en").abstained


class TestCoverageJudge:
    @pytest.mark.parametrize("replies,expected", [
        (("PARTIAL", "NO"), "PARTIAL"), (("full.",), "FULL"), (("UNRELATED",), "UNRELATED"),
        (("I think it is partly answered",), "UNSUPPORTED"), (("",), "UNSUPPORTED"),
    ])
    def test_labels_are_parsed_and_anything_else_fails_closed(self, replies, expected):
        assert LLMNLIBackend(_Provider(*replies)).assess_coverage("p", "q", "a", "en") == expected

    @pytest.mark.parametrize("own_case_reply", ["YES", "", "not sure"])
    def test_a_question_about_the_askers_own_case_is_never_answered_partially(self, own_case_reply):
        """'Why was MY claim rejected?' answered with the general rule reads as the reason."""
        judge = LLMNLIBackend(_Provider("PARTIAL", own_case_reply))
        assert judge.assess_coverage("p", "Why was my claim rejected?", "a", "en") == "UNRELATED"

    def test_strict_mode_needs_a_bare_label(self):
        assert LLMNLIBackend(_Provider("Label: PARTIAL"), strict=True).assess_coverage("p", "q", "a", "en") == "UNSUPPORTED"
        assert LLMNLIBackend(_Provider("PARTIAL", "NO"), strict=True).assess_coverage("p", "q", "a", "en") == "PARTIAL"
        assert LLMNLIBackend(_Provider("PARTIAL", "No, it does not"), strict=True).assess_coverage(
            "p", "q", "a", "en") == "UNRELATED"

    def test_provider_failure_fails_closed(self):
        class Down(_Provider):
            def generate(self, request):
                raise RuntimeError("down")

        assert LLMNLIBackend(Down()).assess_coverage("p", "q", "a", "en") == "UNSUPPORTED"


def test_partial_notes_exist_for_every_language():
    assert set(PARTIAL_NOTES) == {"en", "hi", "ta"}
    assert all(DraftGenerator.partial_note(lang) == note for lang, note in PARTIAL_NOTES.items())

"""A supported general rule is not an answer to a question about the asker's own case.

Found on 15 October 2026: after the prompt began asking the generator to answer the
covered part of a question, "How much fine do I owe for my overdue book?" was answered
with the general rule ("2 rupees per book per day") in 8 of 9 runs. Every claim was
supported and the whole answer was entailed, so the partial-answer path, which held the
personal-case check, was never reached. The check now runs for every released answer.
"""
from __future__ import annotations

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
from pramana.generation.drafting import DraftGenerator, DraftingPolicy
from pramana.ingestion.chunking import chunk_text
from pramana.pipeline import PramanaPipeline
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever

POLICY = "A late return is fined 2 rupees per book for each day."
RULE = "A late return is fined 2 rupees per book for each day."
ENTAILED = NLIScores(0.9, 0.05, 0.05)


class _Provider:
    name = "fixed"

    def __init__(self, *replies: str):
        self.queue = list(replies)

    def generate(self, request):
        text = self.queue.pop(0) if self.queue else ""
        return GenerationResponse(completions=[Completion(text)], model="m", provider=self.name)

    def capabilities(self, model):
        return set()


class _Judge:
    def __init__(self, own_records: bool):
        self.inner = KeywordNLIBackend()
        self.own_records = own_records
        self.asked = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def score_answer(self, premise, question, answer, language):
        return ENTAILED

    def assess_coverage(self, premise, question, answer, language):
        return "FULL"

    def needs_own_records(self, premise, question, answer, language):
        self.asked += 1
        return self.own_records


def _pipeline(judge, *, fail_closed: bool = True) -> PramanaPipeline:
    retriever = MultilingualRetriever()
    index = HybridRetriever(language="en", top_k=3)
    index.add(chunk_text(POLICY, doc_id="rules", language="en"))
    retriever.add_language("en", index)
    provider = _Provider(RULE)
    return PramanaPipeline(
        retriever=retriever, generator=DraftGenerator(provider, DraftingPolicy(n_samples=0)),
        decomposer=RuleBasedDecomposer(), verifier=GroundingVerifier(backend=judge),
        confidence=ConfidenceModel(), policy=CorrectionPolicy(max_iterations=0),
        executor=CorrectionExecutor(provider=provider), fail_closed=fail_closed,
    )


def test_general_rule_is_withheld_for_a_question_about_own_records():
    judge = _Judge(own_records=True)
    result = _pipeline(judge).run("How much fine do I owe for my overdue book?", language="en")
    assert result.abstained and result.stop_reason == "personal_case"
    assert result.final_answer == DraftGenerator.abstention_message("en")
    assert judge.asked == 1


def test_general_question_with_i_is_still_answered():
    judge = _Judge(own_records=False)
    result = _pipeline(judge).run("What is the fine if I return a book late?", language="en")
    assert not result.abstained and result.final_answer == RULE


def test_audit_mode_only_reports():
    judge = _Judge(own_records=True)
    pipeline = _pipeline(judge)
    result = pipeline.run("How much fine do I owe for my overdue book?", language="en", assurance=False)
    assert not result.abstained and result.stop_reason == "audit_only"


def test_without_fail_closed_the_check_does_not_run():
    judge = _Judge(own_records=True)
    result = _pipeline(judge, fail_closed=False).run("How much fine do I owe for my overdue book?", language="en")
    assert not result.abstained and judge.asked == 0


class _Scripted:
    """Provider for the LLM judge: one reply per call, in order."""

    name = "judge"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return GenerationResponse(completions=[Completion(self.replies.pop(0))], model="j", provider=self.name)


def test_llm_backend_asks_coverage_only_for_personal_questions():
    general = _Scripted("NO")
    backend = LLMNLIBackend(provider=general, model="j")
    assert backend.needs_own_records(POLICY, "What is the late fine?", RULE, "en") is False
    assert general.calls == 1

    personal = _Scripted("YES", "NO")
    backend = LLMNLIBackend(provider=personal, model="j")
    assert backend.needs_own_records(POLICY, "How much fine do I owe?", RULE, "en") is True
    assert personal.calls == 2

    described = _Scripted("YES", "YES")
    backend = LLMNLIBackend(provider=described, model="j")
    assert backend.needs_own_records("Your fine on 3 May is 14 rupees.", "How much fine do I owe?",
                                     "You owe 14 rupees.", "en") is False


def test_a_full_coverage_label_no_longer_releases_a_personal_question():
    """24-25 October 2026: the coverage judge called a general rule FULL, so a personal
    question was released. The premise question decides now; coverage is not asked."""
    judge = _Scripted("YES", "NO")
    backend = LLMNLIBackend(provider=judge, model="j")
    assert backend.needs_own_records(POLICY, "How much fine do I owe?", RULE, "en") is True


def test_failed_premise_check_keeps_the_answer_withheld():
    class _Failing(_Scripted):
        def generate(self, request):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("provider down")
            return super().generate(request)

    backend = LLMNLIBackend(provider=_Failing("YES", "unused"), model="j")
    assert backend.needs_own_records(POLICY, "How much fine do I owe?", RULE, "en") is True


def test_non_latin_question_is_asked_again_in_english():
    """22 October 2026: NO for the Tamil form of a personal question, YES in English."""
    judge = _Scripted("NO", "How much fine must I pay for the book I returned late?", "YES")
    backend = LLMNLIBackend(provider=judge, model="j")
    assert backend._asks_about_own_case("நான் எவ்வளவு அபராதம் செலுத்த வேண்டும்?") is True
    assert judge.calls == 3


def test_non_latin_general_question_stays_general():
    judge = _Scripted("NO", "What is the fine for a late return?", "NO")
    backend = LLMNLIBackend(provider=judge, model="j")
    assert backend._asks_about_own_case("தாமதமாகத் திருப்பினால் அபராதம் எவ்வளவு?") is False


def test_failed_translation_keeps_the_question_personal():
    judge = _Scripted("NO", "இது மொழிபெயர்ப்பு அல்ல")
    backend = LLMNLIBackend(provider=judge, model="j")
    assert backend._asks_about_own_case("தாமதமாகத் திருப்பினால் அபராதம் எவ்வளவு?") is True


def test_english_question_is_asked_once():
    judge = _Scripted("NO")
    backend = LLMNLIBackend(provider=judge, model="j")
    assert backend._asks_about_own_case("What is the late fine?") is False and judge.calls == 1

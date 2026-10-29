"""Words of another script inside an answer.

Pilot marks of 13 October 2026 found ten answers in 174 that were correct but carried
Devanagari inside a Tamil or English answer, Tamil inside a Hindi one, or a stray
Korean or Bengali word. The claim checker compares meaning, so it passed them. The
generator now regenerates once and the pipeline withholds what stays impure.
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
from pramana.pipeline import PramanaPipeline
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever
from pramana.schemas import RetrievalResult
from pramana.utils.script import foreign_scripts, is_script_clean, language_instruction

POLICY = "The late fine is 2 rupees per book for each day."
QUESTION = "What is the late fine per book?"


class TestForeignScripts:
    @pytest.mark.parametrize("text,language", [
        ("The late fine is 2 rupees per book per day.", "en"),
        ("देर से लौटाने पर प्रति दिन 2 रुपये का जुर्माना लगता है।", "hi"),
        ("पॉलिसी PX-4471 में ओपीडी कवर है, ₹25,000 तक।", "hi"),           # Latin names and codes are fine
        ("தாமதமாக திருப்பினால் 2 ரூபாய் அபராதம்.", "ta"),
        ("PX-4471 பாலிசி 25,000 ரூபாய் வரை.", "ta"),
        ("Fine: 2 அபராதம்", "ta"),                                  # narrow no-break space is not a letter
    ])
    def test_clean_answers(self, text, language):
        assert is_script_clean(text, language), foreign_scripts(text, language)

    @pytest.mark.parametrize("text,language,script", [
        ("The fine is 2 रुपये per book.", "en", "DEVANAGARI"),
        ("பெரியவர்களுக்கான கட்டணம் 600 रुपये.", "ta", "DEVANAGARI"),
        ("पॉलिसी के लिए (외래 환자 치과 치료) 25,000 रुपये।", "hi", "HANGUL"),
        ("पहला वाक्य (உரிமைகோரல்கள்) दूसरा।", "hi", "TAMIL"),
        ("மகப்பேறு প্রতি வருடம்", "ta", "BENGALI"),
        ("The fine is 2 rupees 罚款.", "en", "CJK"),
    ])
    def test_foreign_scripts_are_named(self, text, language, script):
        assert script in foreign_scripts(text, language)

    def test_danda_and_punctuation_are_not_letters(self):
        assert is_script_clean("यह एक वाक्य है। यह दूसरा है॥", "hi")
        assert is_script_clean("இது ஒரு வாக்கியம்.", "ta")
        assert is_script_clean("It costs 2 (two) rupees; no refund!", "en")

    def test_retry_instruction_names_the_offending_scripts(self):
        text = language_instruction("ta", {"DEVANAGARI": 4})
        assert "Tamil" in text and "DEVANAGARI" in text
        assert "DEVANAGARI" not in language_instruction("ta")


class _Provider:
    name = "fixed"

    def __init__(self, *replies: str):
        self.queue = list(replies)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        text = self.queue.pop(0) if self.queue else ""
        return GenerationResponse(completions=[Completion(text)], model="m", provider=self.name)

    def capabilities(self, model):
        return set()


def _retrieval() -> RetrievalResult:
    index = HybridRetriever(language="en", top_k=3)
    index.add(chunk_text(POLICY, doc_id="policy", language="en"))
    retriever = MultilingualRetriever()
    retriever.add_language("en", index)
    return retriever.retrieve(QUESTION, language="en", top_k=3)


class TestGenerator:
    def test_impure_answer_is_regenerated_once_with_the_scripts_named(self):
        provider = _Provider("The fine is 2 रुपये per book per day.", "The fine is 2 rupees per book per day.")
        draft = DraftGenerator(provider, DraftingPolicy(n_samples=0)).generate(QUESTION, _retrieval(), "en")
        assert draft.text == "The fine is 2 rupees per book per day."
        assert len(provider.requests) == 2
        assert "DEVANAGARI" in provider.requests[1].messages[0].content
        assert "DEVANAGARI" not in provider.requests[0].messages[0].content

    def test_answer_still_impure_after_regeneration_is_rewritten_without_context(self):
        provider = _Provider("அபராதம் 2 रुपये.", "அபராதம் 2 रुपये ஒரு நாளுக்கு.", "அபராதம் ஒரு நாளுக்கு 2 ரூபாய்.")
        draft = DraftGenerator(provider, DraftingPolicy(n_samples=0)).generate(QUESTION, _retrieval(), "ta")
        assert draft.text == "அபராதம் ஒரு நாளுக்கு 2 ரூபாய்."
        assert len(provider.requests) == 3
        rewrite = provider.requests[2]
        assert POLICY not in rewrite.messages[1].content  # the context is left out
        assert "Tamil" in rewrite.messages[0].content

    def test_failed_rewrite_keeps_the_answer_for_the_pipeline_to_withhold(self):
        provider = _Provider("Fine 2 रुपये.", "Fine 2 रुपये.", "Fine 2 रुपये again.")
        draft = DraftGenerator(provider, DraftingPolicy(n_samples=0)).generate(QUESTION, _retrieval(), "en")
        assert draft.text == "Fine 2 रुपये."

    def test_clean_answer_costs_no_extra_call(self):
        provider = _Provider("The fine is 2 rupees per book per day.")
        DraftGenerator(provider, DraftingPolicy(n_samples=0)).generate(QUESTION, _retrieval(), "en")
        assert len(provider.requests) == 1

    def test_abstention_is_not_regenerated(self):
        provider = _Provider("INSUFFICIENT_EVIDENCE")
        DraftGenerator(provider, DraftingPolicy(n_samples=0)).generate(QUESTION, _retrieval(), "en")
        assert len(provider.requests) == 1

    def test_prompt_asks_for_the_covered_parts_only_and_names_the_language(self):
        system = DraftGenerator.build_messages("q", "ctx", "ta")[0].content
        assert "only the parts the context covers" in system and "Tamil" in system


def _pipeline(provider) -> PramanaPipeline:
    retriever = MultilingualRetriever()
    index = HybridRetriever(language="en", top_k=3)
    index.add(chunk_text(POLICY, doc_id="policy", language="en"))
    retriever.add_language("en", index)
    return PramanaPipeline(
        retriever=retriever, generator=DraftGenerator(provider, DraftingPolicy(n_samples=0)),
        decomposer=RuleBasedDecomposer(), verifier=GroundingVerifier(backend=KeywordNLIBackend()),
        confidence=ConfidenceModel(), policy=CorrectionPolicy(max_iterations=0),
        executor=CorrectionExecutor(provider=provider), fail_closed=False,
    )


class TestPipeline:
    def test_answer_that_stays_impure_is_withheld(self):
        impure = "The late fine is 2 rupees per book for each day रुपये."
        result = _pipeline(_Provider(impure, impure)).run(QUESTION, language="en")
        assert result.abstained and result.stop_reason == "language_mismatch"
        assert result.final_answer == DraftGenerator.abstention_message("en")

    def test_regeneration_repairs_the_answer_before_verification(self):
        provider = _Provider("The late fine is 2 रुपये per book for each day.",
                             "The late fine is 2 rupees per book for each day.")
        result = _pipeline(provider).run(QUESTION, language="en")
        assert not result.abstained and "रुपये" not in result.final_answer

    def test_audit_never_withholds_for_script(self):
        # /v1/verify audits somebody else's answer; it reports verdicts and changes nothing.
        pipeline = _pipeline(_Provider("INSUFFICIENT_EVIDENCE"))
        from pramana.schemas import Draft
        draft = Draft(text="The late fine is 2 रुपये per book for each day.", language="en")
        detection = pipeline.verify_answer(draft, _retrieval(), "en", QUESTION)
        assert detection.n_claims >= 1


class TestPruneTidy:
    """Pruning the clause after a connective must not leave the connective behind."""

    @pytest.mark.parametrize("raw,clean", [
        ("You may borrow up to 5 books at a time, and .", "You may borrow up to 5 books at a time."),
        ("You may borrow up to 5 books at a time, and", "You may borrow up to 5 books at a time"),
        ("एक सदस्य अधिकतम 5 पुस्तकें ले सकता है, और", "एक सदस्य अधिकतम 5 पुस्तकें ले सकता है"),
        ("एक सदस्य अधिकतम 5 पुस्तकें ले सकता है, और ।", "एक सदस्य अधिकतम 5 पुस्तकें ले सकता है।"),
        ("அதிகபட்சம் 5 புத்தகங்கள் எடுக்கலாம், மற்றும் .", "அதிகபட்சம் 5 புத்தகங்கள் எடுக்கலாம்."),
        ("The fee is 600 rupees; .", "The fee is 600 rupees."),
    ])
    def test_dangling_connectives_are_removed(self, raw, clean):
        from pramana.correction.policy import _tidy
        assert _tidy(raw) == clean

    @pytest.mark.parametrize("text", [
        "The fine is charged per day and per book.",
        "शुल्क आधार पर।",
        "Rules apply to books and magazines.",
        "Fees are 600 rupees or more.",
    ])
    def test_ordinary_sentences_are_unchanged(self, text):
        from pramana.correction.policy import _tidy
        assert _tidy(text) == text

    def test_prune_of_a_trailing_clause_leaves_a_clean_sentence(self):
        from pramana.correction.policy import prune_claims
        from pramana.schemas import Claim, ClaimVerdict, DetectionResult, Verdict

        text = "You may borrow up to 5 books at a time, and magazines are not included in that limit."
        cut = text.index("magazines")
        detection = DetectionResult([
            ClaimVerdict(claim=Claim("c0", "You may borrow up to 5 books at a time.", "en", source_span=(0, cut - 6)),
                         verdict=Verdict.SUPPORTED, entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05),
            ClaimVerdict(claim=Claim("c1", "Magazines are not included in that limit.", "en",
                                     source_span=(cut, len(text) - 1)),
                         verdict=Verdict.UNVERIFIABLE, entailment_prob=0.05, contradiction_prob=0.05, neutral_prob=0.9),
        ])
        assert prune_claims(text, detection) == "You may borrow up to 5 books at a time."


class TestMixedWords:
    """Latin is allowed in Hindi and Tamil answers, but not inside one word (20 October 2026)."""

    @pytest.mark.parametrize("text,language", [
        ("पॉolicy PX-4471 में 25,000 रुपए तक कवर है।", "hi"),
        ("नiroki kiye gaye daave", "hi"),
        ("அதிகபட்சம் 5 புத்தகgal", "ta"),
    ])
    def test_word_mixing_scripts_is_flagged(self, text, language):
        assert foreign_scripts(text, language).get("MIXED_WORD")

    @pytest.mark.parametrize("text,language", [
        ("Lakeview Library-ன் விதிகளின்படி 5 புத்தகங்கள் எடுக்கலாம்.", "ta"),
        ("कर्मचारी 15 unused leave days ले जा सकते हैं।", "hi"),
        ("पॉलिसी PX-4471 में ओपीडी कवर है।", "hi"),
        ("The fee is 600 rupees.", "en"),
    ])
    def test_separate_latin_words_are_allowed(self, text, language):
        assert "MIXED_WORD" not in foreign_scripts(text, language)

    def test_retry_instruction_explains_mixed_words(self):
        text = language_instruction("hi", {"MIXED_WORD": 1})
        assert "one script" in text and "MIXED_WORD" not in text


def test_prompt_asks_for_the_token_instead_of_a_missing_information_sentence():
    """22 October 2026: 'the context gives no information about the children's fee' was released."""
    system = DraftGenerator.build_messages("q", "ctx", "hi")[0].content
    assert "INSUFFICIENT_EVIDENCE instead of a sentence" in system

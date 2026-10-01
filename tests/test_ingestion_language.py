"""Tests for language and script identification.

The failure this guards against is not a wrong label in a log: if a Tamil
document is filed as Hindi, it is embedded into the wrong index and every
per-language retrieval number for the rest of the project is wrong, with nothing
to indicate it.
"""

from __future__ import annotations

import pytest

from pramana.ingestion.language import (
    ROMAN_HINDI_MARKERS,
    assert_language,
    count_scripts,
    detect,
    detect_language,
    detect_script,
    normalize,
    strip_punctuation,
)

HINDI = "मेरा क्लेम क्यों रिजेक्ट हुआ? कृपया बताइए।"
TAMIL = "எனது காப்பீட்டு உரிமைகோரல் ஏன் நிராகரிக்கப்பட்டது?"
ENGLISH = "Why was my health insurance claim rejected last month?"
ROMAN_HI = "mera claim kyun reject hua hai bhai, kuch samajh nahi aa raha"
ROMAN_TA = "enaku enna panna mudiyum, enoda claim eppadi irukku"


class TestNormalisation:
    def test_collapses_whitespace(self):
        assert normalize("  a\n\n b\t c  ") == "a b c"

    def test_nfc_makes_equivalent_encodings_identical(self):
        """Composed and decomposed Devanagari look identical but hash differently.
        Without NFC the same string can miss an exact-match lookup and inflate the
        apparent hallucination rate.

        Written as escapes, not literals: a source file saved in either normal
        form would make the two sides equal before the test even runs.
        """
        composed = "क़"                 # KA WITH NUKTA, precomposed
        decomposed = "क़"         # KA + combining NUKTA
        assert composed != decomposed, "inputs must differ before normalisation"
        assert normalize(composed) == normalize(decomposed)

    def test_strips_zero_width_characters(self):
        assert normalize("ab​c") == "abc"

    def test_punctuation_stripping_preserves_indic_marks(self):
        """A naive [^\\w\\s] filter removes Devanagari viramas and Tamil vowel
        signs, silently changing the words themselves."""
        text = "क्या, हुआ?"
        out = strip_punctuation(text)
        assert "," not in out and "?" not in out
        assert "्" in out, "virama must survive"

    def test_tamil_vowel_signs_survive_punctuation_stripping(self):
        out = strip_punctuation("நிராகரிக்கப்பட்டது!")
        assert "!" not in out
        assert "ி" in out and "்" in out


class TestScriptCounting:
    def test_counts_are_disjoint_per_script(self):
        c = count_scripts(HINDI)
        assert c["devanagari"] > 0 and c["tamil"] == 0 and c["latin"] == 0

        c = count_scripts(TAMIL)
        assert c["tamil"] > 0 and c["devanagari"] == 0

    def test_detects_native_roman_and_mixed(self):
        assert detect_script(HINDI) == "native"
        assert detect_script(TAMIL) == "native"
        assert detect_script(ENGLISH) == "roman"
        assert detect_script("मेरा claim reject हुआ क्यों बताओ") == "mixed"

    def test_stray_latin_does_not_make_a_document_code_mixed(self):
        """A product code or acronym inside a long Hindi passage is not
        code-mixing; treating it as such would fragment the corpus."""
        assert detect_script(HINDI * 4 + " ID42") == "native"


class TestNativeScriptDetection:
    @pytest.mark.parametrize(
        ("text", "language"),
        [(HINDI, "hi"), (TAMIL, "ta"), (ENGLISH, "en")],
    )
    def test_identifies_target_languages(self, text, language):
        assert detect_language(text).language == language

    def test_native_script_detection_is_high_confidence(self):
        """Unicode ranges are disjoint, so this is near-exact -- and it must score
        far above the vocabulary heuristic used for Romanised text."""
        assert detect_language(HINDI).confidence > 0.9
        assert detect_language(TAMIL).confidence > 0.9

    def test_hindi_and_tamil_are_never_confused(self):
        assert detect_language(HINDI).language != "ta"
        assert detect_language(TAMIL).language != "hi"

    def test_convenience_wrapper_returns_both_labels(self):
        assert detect(TAMIL) == ("ta", "native")


class TestRomanisedDetection:
    def test_detects_romanised_hindi(self):
        d = detect_language(ROMAN_HI)
        assert d.language == "hi"
        assert d.script == "roman"

    def test_detects_romanised_tamil(self):
        d = detect_language(ROMAN_TA)
        assert d.language == "ta"
        assert d.script == "roman"

    def test_romanised_confidence_is_capped_below_native(self):
        """This is a vocabulary heuristic. Overstating its confidence would let
        unreviewed data into the evaluation set."""
        roman = detect_language(ROMAN_HI).confidence
        native = detect_language(HINDI).confidence
        assert roman < native
        assert roman <= 0.80

    def test_romanised_detection_is_flagged_for_review(self):
        assert detect_language(ROMAN_HI).needs_review

    def test_plain_english_is_not_misread_as_romanised_hindi(self):
        """The marker list deliberately omits ambiguous words. If 'me', 'to',
        'is' or 'in' crept in, ordinary English would be labelled Hindi."""
        for text in [
            ENGLISH,
            "Please send me the policy document to my email address.",
            "The claim is in review and it has to be approved by the manager.",
            "This is the information you requested about your account.",
        ]:
            assert detect_language(text).language == "en", text

    def test_marker_list_excludes_common_english_words(self):
        ambiguous = {"me", "to", "he", "us", "is", "in", "the", "a", "an", "so", "no", "on"}
        assert not (ROMAN_HINDI_MARKERS & ambiguous), (
            f"ambiguous English words in the Hindi marker list: {ROMAN_HINDI_MARKERS & ambiguous}"
        )


class TestEdgeCases:
    def test_empty_input_is_flagged_not_crashed(self):
        d = detect_language("")
        assert d.confidence == 0.0 and d.needs_review

    def test_very_short_romanised_input_is_low_confidence(self):
        """'ok' or 'claim status' carries no language cue. Claiming confidence
        here would be dishonest, so it falls back with a warning."""
        d = detect_language("ok")
        assert d.confidence < 0.5 and d.needs_review

    def test_digits_and_symbols_do_not_crash(self):
        assert detect_language("12345 !!! ###").language in {"en", "hi", "ta"}

    def test_non_target_indic_script_is_warned_about(self):
        """Bengali must not be silently filed as Hindi."""
        d = detect_language("আমার দাবি কেন প্রত্যাখ্যান করা হয়েছে")
        assert d.warnings and any("outside" in w for w in d.warnings)

    def test_code_mixed_text_is_marked(self):
        d = detect_language("मेरा claim reject हो गया, please check karo")
        assert d.is_code_mixed and d.script == "mixed"


class TestCorpusGuard:
    def test_accepts_matching_language(self):
        assert_language(HINDI, "hi")
        assert_language(TAMIL, "ta")

    def test_rejects_mismatched_language(self):
        """Loading an English paragraph into the Hindi corpus would corrupt
        per-language retrieval metrics invisibly. Fail at load time instead."""
        with pytest.raises(ValueError, match="expected hi"):
            assert_language(ENGLISH, "hi")

    def test_rejects_tamil_filed_as_hindi(self):
        with pytest.raises(ValueError, match="expected hi, detected ta"):
            assert_language(TAMIL, "hi")

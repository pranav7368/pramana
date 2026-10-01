"""Language and script identification for English, Hindi and Tamil.

**Why this module is not optional.** Real users do not type the way benchmarks
assume. They write Hindi in Devanagari *and* in Roman letters, they code-mix
freely with English, and the same language in a different script provokes
measurably different model behaviour (arXiv:2512.10780, "Script Gap"). A pipeline
that assumes one script per language works on a test set and fails on live
traffic.

Implemented with Unicode block statistics rather than a trained classifier:

* Devanagari and Tamil occupy disjoint Unicode ranges, so script detection on
  native text is exact -- a statistical model can only be worse.
* No model download, no RAM cost, deterministic, and it works on the two-word
  fragments where trained language-ID models are least reliable.

The genuinely hard case is *Romanised* Hindi, where the script carries no signal
and only vocabulary can decide. That is handled by function-word matching, whose
limits are documented in ``detect_language``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pramana.schemas import Language, Script

# ── Unicode ranges ────────────────────────────────────────────────────────────

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
TAMIL = re.compile(r"[஀-௿]")
LATIN = re.compile(r"[A-Za-z]")

# Other Indic blocks: not target languages, but detecting them prevents a Bengali
# or Telugu document being silently filed as Hindi.
OTHER_INDIC = re.compile(r"[ঀ-୿ఀ-ൿ਀-੿]")

_WHITESPACE = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮﻿]")

# ── Romanised-Hindi cues ──────────────────────────────────────────────────────
#
# High-frequency Hindi function words that are rare or absent as English words.
#
# EXCLUDED ON PURPOSE, despite being perfectly good Hindi: "the" (थे), "to" (तो),
# "hi" (ही), "mat" (मत), "par" (पर). Each is also a common English word, and an
# early version that included them classified
#   "Please send me the policy document to my email address"  ->  Hindi.
# English is the majority language in this corpus, so a false positive on English
# costs more than a missed Romanised-Hindi cue. Recall is recovered from the many
# unambiguous markers that remain.
ROMAN_HINDI_MARKERS: frozenset[str] = frozenset([
    "kya", "kyun", "kyo", "kaise", "kahan", "kaha", "kab",
    "kaun", "kitna", "kitne", "kitni", "mera", "meri", "mere",
    "tera", "teri", "tumhara", "aapka", "apna", "apne", "unka",
    "uska", "iska", "hai", "hain", "tha", "thi", "hoga",
    "hogi", "honge", "raha", "rahi", "rahe", "gaya", "gayi",
    "gaye", "nahi", "nahin", "bina", "sirf", "bhi", "bhai",
    "yaar", "acha", "thik", "aur", "lekin", "magar", "kyunki",
    "isliye", "phir", "abhi", "karna", "karta", "karti", "karte",
    "kiya", "kiye", "karo", "dena", "dete", "milta", "milti",
    "jata", "jati", "jaye", "mujhe", "tujhe", "hume", "unhe",
    "inhe", "uske", "iske", "jiske", "mein", "wala", "wali",
    "wale", "hota", "hoti", "hote",
])

ROMAN_TAMIL_MARKERS: frozenset[str] = frozenset([
    "enna", "eppadi", "enge", "eppo", "yaar", "evlo", "ethana",
    "enaku", "unaku", "avanuku", "enoda", "unoda", "avanoda", "irukku",
    "illa", "illai", "venum", "vendam", "mudiyum", "mudiyathu", "naan",
    "nee", "avan", "aval", "naanga", "neenga", "avanga", "pannu",
    "panren", "panra", "sollu", "solren", "vandhu", "poyi", "ithu",
    "athu", "inge", "ange", "romba", "konjam", "nalla",
])

_WORD = re.compile(r"[a-z]+")


@dataclass(slots=True)
class LanguageDetection:
    language: Language
    script: Script
    confidence: float
    """Rough reliability of the call. Native-script detections are near-certain;
    Romanised detections rest on vocabulary cues and score lower."""

    char_counts: dict[str, int]
    is_code_mixed: bool = False
    warnings: list[str] | None = None

    @property
    def needs_review(self) -> bool:
        """Whether a human should check this call before it enters an eval set."""
        return self.confidence < 0.6 or bool(self.warnings)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Canonicalise text before any comparison, hashing or embedding.

    NFC matters for Indic scripts specifically: the same visible Devanagari or
    Tamil string can be encoded with precomposed or decomposed sequences, and
    without normalisation two identical-looking strings hash differently, break
    exact-match retrieval, and silently inflate apparent hallucination rates.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def strip_punctuation(text: str) -> str:
    """Remove punctuation while preserving Indic combining marks.

    A naive ``[^\\w\\s]`` filter strips Devanagari viramas and Tamil vowel signs,
    which changes the words themselves.
    """
    return "".join(
        ch for ch in text if not unicodedata.category(ch).startswith("P") or ch in "-'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────


def count_scripts(text: str) -> dict[str, int]:
    return {
        "devanagari": len(DEVANAGARI.findall(text)),
        "tamil": len(TAMIL.findall(text)),
        "latin": len(LATIN.findall(text)),
        "other_indic": len(OTHER_INDIC.findall(text)),
    }


def detect_script(text: str) -> Script:
    c = count_scripts(text)
    indic = c["devanagari"] + c["tamil"]
    if indic and c["latin"]:
        # A few stray Latin characters (a product code, an acronym) do not make
        # a document code-mixed; a substantial share does.
        minority = min(indic, c["latin"]) / (indic + c["latin"])
        return "mixed" if minority > 0.15 else ("native" if indic > c["latin"] else "roman")
    if indic:
        return "native"
    return "roman"


def _roman_marker_ratio(text: str, markers: frozenset[str]) -> tuple[float, int]:
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0, 0
    hits = sum(1 for w in words if w in markers)
    return hits / len(words), hits


def detect_language(text: str, *, default: Language = "en") -> LanguageDetection:
    """Identify language and script.

    Native-script text is decided by Unicode range and is effectively exact.
    Romanised text is decided by function-word frequency, which is genuinely
    approximate -- short inputs ("ok", "claim status") carry no cue at all and
    fall back to ``default`` with low confidence. Callers doing anything that
    matters should check ``needs_review`` rather than trusting the label blindly.
    """
    text = normalize(text)
    counts = count_scripts(text)
    warnings: list[str] = []

    if not text:
        return LanguageDetection(default, "roman", 0.0, counts, warnings=["empty input"])

    if counts["other_indic"] > counts["devanagari"] + counts["tamil"]:
        warnings.append(
            "text is predominantly an Indic script outside {Devanagari, Tamil} — "
            "not a target language of this project"
        )

    script = detect_script(text)
    indic_total = counts["devanagari"] + counts["tamil"]

    # ── native script: decisive ────────────────────────────────────────────────
    if indic_total > 0:
        if counts["tamil"] > counts["devanagari"]:
            language: Language = "ta"
            purity = counts["tamil"] / indic_total
        else:
            language = "hi"
            purity = counts["devanagari"] / indic_total

        code_mixed = script == "mixed"
        confidence = min(0.99, 0.75 + 0.24 * purity)
        if code_mixed:
            confidence *= 0.9
            warnings.append("code-mixed with Latin script")
        return LanguageDetection(language, script, confidence, counts, code_mixed, warnings or None)

    # ── Latin script: vocabulary is the only evidence ─────────────────────────
    hi_ratio, hi_hits = _roman_marker_ratio(text, ROMAN_HINDI_MARKERS)
    ta_ratio, ta_hits = _roman_marker_ratio(text, ROMAN_TAMIL_MARKERS)
    n_words = len(_WORD.findall(text.lower()))

    if n_words < 3:
        warnings.append(f"only {n_words} word(s) — too short to identify Romanised text")
        return LanguageDetection(default, "roman", 0.25, counts, warnings=warnings)

    if max(hi_ratio, ta_ratio) >= 0.15 and max(hi_hits, ta_hits) >= 2:
        language = "hi" if hi_ratio >= ta_ratio else "ta"
        ratio = max(hi_ratio, ta_ratio)
        # Capped well below native-script confidence: this is a vocabulary
        # heuristic, and overstating it would let unreviewed data into the
        # evaluation set.
        return LanguageDetection(
            language,
            "roman",
            min(0.80, 0.45 + ratio),
            counts,
            is_code_mixed=ratio < 0.5,
            warnings=["Romanised Indic detected by vocabulary — verify before use in eval data"],
        )

    return LanguageDetection("en", "roman", 0.85 if n_words >= 5 else 0.6, counts)


def detect(text: str, *, default: Language = "en") -> tuple[Language, Script]:
    """Convenience wrapper returning just the labels."""
    d = detect_language(text, default=default)
    return d.language, d.script


# ──────────────────────────────────────────────────────────────────────────────
# Corpus-level checks
# ──────────────────────────────────────────────────────────────────────────────


def assert_language(text: str, expected: Language, *, min_confidence: float = 0.6) -> None:
    """Raise if ``text`` is not confidently in ``expected``.

    Used when loading the corpus. A Hindi file that silently contains English
    paragraphs would corrupt per-language retrieval metrics in a way that is very
    hard to notice later -- far better to fail at load time.
    """
    d = detect_language(text)
    if d.language != expected or d.confidence < min_confidence:
        raise ValueError(
            f"expected {expected}, detected {d.language} (script={d.script}, "
            f"confidence={d.confidence:.2f}). Sample: {text[:80]!r}"
        )


def script_profile(text: str) -> str:
    """One-line diagnostic, e.g. ``hi/native conf=0.98 dev=142 lat=3``."""
    d = detect_language(text)
    c = d.char_counts
    return (
        f"{d.language}/{d.script} conf={d.confidence:.2f} "
        f"dev={c['devanagari']} tam={c['tamil']} lat={c['latin']}"
        + (" [review]" if d.needs_review else "")
    )

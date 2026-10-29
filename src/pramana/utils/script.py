"""Script hygiene for generated answers.

A claim checker compares meaning, so an answer that is correct but carries a
Devanagari word inside a Tamil sentence, or a Korean phrase inside a Hindi one,
passes verification. Pilot marks of 13 October 2026 found ten such answers in
174 (``docs/08_RESULTS.md``, P9). All came from cross-language questions, where the
model copies source-language words into an answer written in another language.

The rule is deliberately narrow. Letters of the answer's own script are allowed,
and so are Latin letters for every language, because Hindi and Tamil text
routinely keeps English words, product names and codes in Latin. Anything else
is foreign: Devanagari in a Tamil or English answer, Tamil in a Hindi or English
answer, and every script that is none of the three (Hangul, Bengali, CJK, ...).
Digits, punctuation, currency signs and combining marks of an allowed script are
never counted.
"""

from __future__ import annotations

import re
import unicodedata

from pramana.schemas import Language

# Script names as they appear in Unicode character names.
_OWN_SCRIPT: dict[str, str] = {"en": "LATIN", "hi": "DEVANAGARI", "ta": "TAMIL"}
_LANGUAGE_NAMES: dict[str, str] = {"en": "English", "hi": "Hindi (Devanagari script)", "ta": "Tamil (Tamil script)"}


def _script_of(char: str) -> str | None:
    """Script of a letter or combining mark, or None for anything else."""
    category = unicodedata.category(char)
    if not (category.startswith("L") or category in {"Mn", "Mc"}):
        return None
    try:
        name = unicodedata.name(char)
    except ValueError:
        return None
    return name.split(" ", 1)[0]


def foreign_scripts(text: str, language: Language) -> dict[str, int]:
    """Letters per script that do not belong in an answer written in ``language``."""
    allowed = {_OWN_SCRIPT.get(language, "LATIN"), "LATIN"}
    found: dict[str, int] = {}
    for char in text:
        script = _script_of(char)
        if script is None or script in allowed:
            continue
        # Combining marks are named after their own script too (e.g. COMBINING ...);
        # those belong to the base letter, so only count real letters and script marks.
        if script == "COMBINING":
            continue
        found[script] = found.get(script, 0) + 1
    own = _OWN_SCRIPT.get(language, "LATIN")
    if own != "LATIN":
        # Latin is allowed in Hindi and Tamil for names and codes, but not inside a
        # word of the language's own script: "पॉolicy" and "नiroki" (20 October 2026).
        # Hyphens separate words, so "Library-ன்" stays allowed.
        for word in re.split(r"[\s\-‐‑–—/(),.;:!?।\"'’‘“”]+", text):
            scripts = {_script_of(c) for c in word} - {None, "COMBINING"}
            if own in scripts and "LATIN" in scripts:
                found["MIXED_WORD"] = found.get("MIXED_WORD", 0) + 1
    return found


def is_script_clean(text: str, language: Language) -> bool:
    return not foreign_scripts(text, language)


def language_instruction(language: Language, found: dict[str, int] | None = None) -> str:
    """A reminder appended to the prompt; names the offending scripts on a retry."""
    name = _LANGUAGE_NAMES.get(language, "English")
    text = (
        f"\nWrite the whole answer in {name} only. If the context is in another language, translate the "
        "facts; do not copy words or phrases from it in another script. "
        "Latin letters are allowed only for names, codes and abbreviations."
    )
    scripts = sorted(k for k in (found or {}) if k != "MIXED_WORD")
    if scripts:
        text += (
            " Your previous answer contained " + ", ".join(scripts) + " characters that do not "
            "belong in the answer. Write it again without them."
        )
    if found and "MIXED_WORD" in found:
        text += " Your previous answer had words that mix Latin letters into the language's own script; write each word in one script."
    return text

"""Claim decomposition — break an answer into atomic, verifiable propositions.

Stage 3a of the pipeline, and the first half of the project's contribution.

**Why decompose at all.** An answer-level faithfulness score of 0.6 says something
is wrong but not *what*. The correction policy needs to know which claim to fix,
remove, or abstain on, and a citation is only trustworthy if it attaches to a
specific proposition. The literature also reports claim-level NLI correlating
substantially better with human judgement than sentence-level scoring.

**What makes a good claim.** Three properties, in priority order:

1. **Atomic** — one proposition. "The claim was rejected because it was late"
   is two: that it was rejected, and the reason. They can be independently true.
2. **Self-contained** — no unresolved references. This is where Hindi and Tamil
   diverge from English: both are *pro-drop*, so the subject is routinely absent.
   A decomposer that copies surface text produces "रिजेक्ट हो गया" ("was rejected")
   with no subject, which the verifier then marks UNVERIFIABLE for a grammatical
   reason rather than a factual one — an error that looks like a multilingual
   model weakness and is actually a decomposition bug.
3. **Span-linked** — character offsets into the draft, so PRUNE can excise exactly
   this claim. Necessarily lost when a claim is rewritten to restore an elided
   subject, which is why ``source_span`` is optional.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from pramana.generation.base import GenerationRequest, LLMProvider, Message
from pramana.ingestion.chunking import split_sentences
from pramana.schemas import Claim, Draft, Language

log = logging.getLogger(__name__)


class Decomposer(Protocol):
    def decompose(self, draft: Draft) -> list[Claim]: ...


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = {
    "en": (
        "Split the given answer into atomic factual claims.\n"
        "Rules:\n"
        "1. Each claim states exactly ONE fact.\n"
        "2. Each claim must stand alone: replace every pronoun with the noun it "
        "refers to, and supply any subject that was omitted.\n"
        "3. Preserve numbers, dates and negations exactly. Never drop a 'not'.\n"
        "4. Do not add information that is not in the answer. Do not define or "
        "explain words, and do not write commentary about the answer itself.\n"
        "5. Output one claim per line, no numbering, no commentary.\n"
        "6. If the answer contains no factual claim, output NO_CLAIMS."
    ),
    "hi": (
        "दिए गए उत्तर को अलग-अलग तथ्यों (claims) में तोड़ें।\n"
        "नियम:\n"
        "1. हर claim में केवल एक तथ्य हो।\n"
        "2. हर claim अपने आप में पूरा हो: सर्वनाम की जगह पूरा नाम लिखें, और यदि कर्ता "
        "छूटा हुआ है तो उसे जोड़ें।\n"
        "3. संख्याएँ, तिथियाँ और 'नहीं' बिल्कुल वैसे ही रखें।\n"
        "4. उत्तर में जो नहीं है, वह न जोड़ें। शब्दों की परिभाषा या उत्तर पर टिप्पणी न लिखें।\n"
        "5. हर पंक्ति में एक claim, बिना नंबर, बिना टिप्पणी।\n"
        "6. यदि कोई तथ्य नहीं है तो NO_CLAIMS लिखें।"
    ),
    "ta": (
        "கொடுக்கப்பட்ட பதிலைத் தனித்தனி உண்மைக் கூற்றுகளாகப் பிரிக்கவும்.\n"
        "விதிகள்:\n"
        "1. ஒவ்வொரு கூற்றிலும் ஒரே ஒரு உண்மை மட்டும் இருக்க வேண்டும்.\n"
        "2. ஒவ்வொரு கூற்றும் தனித்து நிற்க வேண்டும்: பிரதிபெயர்களுக்குப் பதிலாக "
        "முழுப் பெயரையும், விடுபட்ட எழுவாயையும் சேர்க்கவும்.\n"
        "3. எண்கள், தேதிகள், எதிர்மறைச் சொற்களை அப்படியே வைக்கவும்.\n"
        "4. பதிலில் இல்லாத தகவலைச் சேர்க்க வேண்டாம். சொற்களுக்கு விளக்கமோ "
        "பதிலைப் பற்றிய கருத்தையோ எழுத வேண்டாம்.\n"
        "5. ஒரு வரிக்கு ஒரு கூற்று; எண்ணிடல் வேண்டாம், விளக்கம் வேண்டாம்.\n"
        "6. உண்மைக் கூற்று எதுவும் இல்லையெனில் NO_CLAIMS என எழுதவும்."
    ),
}

NO_CLAIMS = "NO_CLAIMS"

# Few-shot examples. The Hindi example exists specifically to demonstrate subject
# restoration, which is the behaviour most likely to be dropped.
_FEWSHOT: dict[Language, tuple[str, str]] = {
    "en": (
        "Your claim was rejected because it was not submitted within 30 days of discharge.",
        "The claim was rejected.\nThe claim was not submitted within 30 days of discharge.",
    ),
    "hi": (
        "रिजेक्ट हो गया क्योंकि 30 दिन के अंदर जमा नहीं किया गया।",
        "क्लेम रिजेक्ट हो गया।\nक्लेम 30 दिन के अंदर जमा नहीं किया गया।",
    ),
    "ta": (
        "30 நாட்களுக்குள் சமர்ப்பிக்கப்படாததால் நிராகரிக்கப்பட்டது.",
        "உரிமைகோரல் நிராகரிக்கப்பட்டது.\nஉரிமைகோரல் 30 நாட்களுக்குள் சமர்ப்பிக்கப்படவில்லை.",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based fallback
# ──────────────────────────────────────────────────────────────────────────────

_CONNECTIVES = {
    "en": re.compile(r"\s+(?:because|since|and|but|however|therefore|so that|although)\s+", re.I),
    "hi": re.compile(r"\s+(?:क्योंकि|और|लेकिन|इसलिए|परंतु|तथा)\s+"),
    "ta": re.compile(r"\s+(?:ஏனெனில்|மற்றும்|ஆனால்|எனவே)\s+"),
}


@dataclass(slots=True)
class RuleBasedDecomposer:
    """Sentence- and connective-based splitting. No model, no network.

    A genuine fallback, not a toy: it keeps the pipeline testable offline and
    serves as the **ablation baseline** that isolates how much LLM decomposition
    actually contributes.

    Its known weakness is exactly the interesting one: it cannot restore an elided
    subject, so Hindi and Tamil claims may remain incomplete. That gap is the
    measurement.
    """

    split_connectives: bool = True

    def decompose(self, draft: Draft) -> list[Claim]:
        text = draft.text.strip()
        if not _is_decomposable(text):
            return []

        claims: list[Claim] = []
        cursor = 0

        for sentence in split_sentences(text, draft.language):
            parts = self._split_clauses(sentence, draft.language)
            for part in parts:
                part = part.strip()
                if not _is_substantive(part):
                    continue
                start = text.find(part, cursor)
                span = (start, start + len(part)) if start >= 0 else None
                if start >= 0:
                    cursor = start + len(part)
                claims.append(
                    Claim(
                        claim_id=f"c{len(claims)}",
                        text=part,
                        language=draft.language,
                        source_span=span,
                    )
                )
        return claims

    def _split_clauses(self, sentence: str, language: Language) -> list[str]:
        if not self.split_connectives:
            return [sentence]
        pattern = _CONNECTIVES.get(language)
        if pattern is None:
            return [sentence]
        parts = [p for p in pattern.split(sentence) if p.strip()]
        # Only accept the split if both halves are substantive; otherwise a
        # trailing "and" fragment becomes a claim with no content.
        return parts if len(parts) > 1 and all(_is_substantive(p) for p in parts) else [sentence]


# ──────────────────────────────────────────────────────────────────────────────
# LLM decomposer
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class LLMDecomposer:
    """Prompt-based decomposition with a rule-based safety net.

    Falls back rather than raising when the model returns something unusable: a
    decomposition failure must degrade to coarser claims, not abort verification
    of the whole answer.
    """

    provider: LLMProvider
    model: str = ""
    max_tokens: int = 1024
    strict: bool = False
    fallback: RuleBasedDecomposer = field(default_factory=RuleBasedDecomposer)
    max_claims: int = 20
    """Guard against a model that emits a list instead of claims. An answer with
    more than ~20 atomic claims signals a decomposition failure, not a rich answer."""

    def decompose(self, draft: Draft) -> list[Claim]:
        text = draft.text.strip()
        if not _is_decomposable(text):
            return []

        try:
            raw = self.provider.generate(
                GenerationRequest(
                    messages=self._build_messages(text, draft.language),
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
            ).text
        except Exception as exc:
            if self.strict:
                raise
            log.warning("LLM decomposition failed (%s); using rule-based fallback", type(exc).__name__)
            return self.fallback.decompose(draft)

        claims = self._parse(raw, draft)

        if _is_inflated(claims, text):
            # Observed live 2026-08-10: given the terse Tamil answer "30 நாட்கள், ஆம்."
            # ("30 days, yes.") the model returned "30 days is a specific deadline"
            # and "Yes indicates agreement" -- meta-commentary *about* the answer
            # rather than propositions *in* it. The second was then scored
            # UNVERIFIABLE, reporting a 50% hallucination rate for an answer that
            # was entirely correct. A decomposition artefact must never be counted
            # as a generator failure.
            log.warning(
                "decomposition inflated a %d-char answer into %d claims totalling %d chars; "
                "falling back to rule-based",
                len(text),
                len(claims),
                sum(len(c.text) for c in claims),
            )
            return self.fallback.decompose(draft)

        if not claims and NO_CLAIMS not in raw.upper():
            log.info("LLM returned no usable claims; falling back to rule-based")
            return self.fallback.decompose(draft)
        return claims

    def _build_messages(self, text: str, language: Language) -> tuple[Message, ...]:
        system = _DECOMPOSE_SYSTEM.get(language, _DECOMPOSE_SYSTEM["en"])
        example_in, example_out = _FEWSHOT.get(language, _FEWSHOT["en"])
        return (
            Message("system", system),
            Message("user", f"Answer:\n{example_in}"),
            Message("assistant", example_out),
            Message("user", f"Answer:\n{text}"),
        )

    def _parse(self, raw: str, draft: Draft) -> list[Claim]:
        if NO_CLAIMS in raw.upper():
            return []

        claims: list[Claim] = []
        cursor = 0
        for line in raw.splitlines():
            line = _strip_list_marker(line)
            if not _is_substantive(line) or NO_CLAIMS in line.upper():
                continue

            # Span linking is best-effort: a claim rewritten to restore an elided
            # subject no longer occurs verbatim in the draft, and reporting a
            # wrong span would make PRUNE excise the wrong text.
            start = draft.text.find(line, cursor)
            span = (start, start + len(line)) if start >= 0 else None
            if start >= 0:
                cursor = start + len(line)

            claims.append(
                Claim(
                    claim_id=f"c{len(claims)}",
                    text=line,
                    language=draft.language,
                    source_span=span,
                )
            )
            if len(claims) >= self.max_claims:
                log.warning("decomposition hit the %d-claim cap; using full rule-based answer", self.max_claims)
                return self.fallback.decompose(draft)

        return claims


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_MIN_CLAIM_CHARS = 8
_MIN_DECOMPOSABLE_CHARS = 12
_INFLATION_LIMIT = 2.0
"""Claims may restate an answer more explicitly -- restoring an elided subject
genuinely adds characters -- but total claim text several times the answer's
length means the model is inventing, not decomposing."""

# Abstention markers. An abstention is not an answer and must never enter claim
# verification: decomposing "INSUFFICIENT_EVIDENCE" yields the claim "There is
# insufficient evidence", which is then scored UNVERIFIABLE and counted as a
# hallucination. The system would be penalised for correctly declining to answer.
_ABSTENTION_MARKERS = ("INSUFFICIENT_EVIDENCE", "NO_CLAIMS")


def _is_decomposable(text: str) -> bool:
    if len(text) < _MIN_DECOMPOSABLE_CHARS:
        return False
    upper = text.upper()
    return not any(marker in upper for marker in _ABSTENTION_MARKERS)


def _is_inflated(claims: list[Claim], source: str) -> bool:
    if not claims:
        return False
    return sum(len(c.text) for c in claims) > max(60, len(source) * _INFLATION_LIMIT)


def _strip_list_marker(line: str) -> str:
    return _LIST_MARKER.sub("", line).strip()


def _is_substantive(text: str) -> bool:
    """Whether a fragment carries enough content to be worth verifying.

    Filters bare connectives and stray punctuation. Sending those to the verifier
    produces UNVERIFIABLE verdicts that are real in form and meaningless in
    substance, which would inflate the measured hallucination rate.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CLAIM_CHARS:
        return False
    return any(ch.isalnum() for ch in stripped)


def decomposition_report(claims: list[Claim]) -> str:
    if not claims:
        return "0 claims"
    linked = sum(1 for c in claims if c.source_span is not None)
    return f"{len(claims)} claims · {linked} span-linked · {len(claims) - linked} rewritten"

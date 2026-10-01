"""Offline generator returning fixtures with *planted* hallucinations.

This is what makes the project developable on a 7.7 GB laptop with no GPU and no
internet.  More importantly, because the defect in each fixture is known in
advance, the detection and correction stages become genuinely **testable** —
assertions on expected verdicts, not smoke tests that only check nothing crashed.

Each fixture carries the ground truth the pipeline should recover:

    F01  number altered (30 -> 15 days)      CONTRADICTED   -> REGENERATE
    F02  fact absent from context             UNVERIFIABLE   -> PRUNE
    F03  fully grounded                       SUPPORTED      -> ACCEPT
    F04  empty retrieval, fluent answer       UNVERIFIABLE   -> ABSTAIN
    F05  negation dropped                     CONTRADICTED   -> REGENERATE
    F06  Hindi, pro-drop subject elided       decomposer must restore subject
    F07  Tamil, agglutinated multi-claim word decomposer must split
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from pramana.generation.base import (
    Capability,
    Completion,
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    TokenLogProb,
    Usage,
)

Verdict = str  # "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE"


@dataclass(slots=True)
class Fixture:
    """A canned answer plus the ground truth a correct pipeline must recover."""

    fixture_id: str
    language: str
    query: str
    context: list[str]
    answer: str
    expected_claims: list[str]
    expected_verdicts: list[Verdict]
    expected_action: str
    defect: str
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.expected_claims) != len(self.expected_verdicts):
            raise ValueError(
                f"fixture {self.fixture_id}: {len(self.expected_claims)} claims but "
                f"{len(self.expected_verdicts)} verdicts"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Fixture corpus
# ──────────────────────────────────────────────────────────────────────────────

FIXTURES: list[Fixture] = [
    Fixture(
        fixture_id="F01",
        language="en",
        query="Why was my health insurance claim rejected?",
        context=[
            "Claims are rejected if submitted more than 30 days after the discharge date.",
            "Rejected claims may be appealed within 60 days of the rejection notice.",
        ],
        answer=(
            "Your claim was rejected because it was not submitted within 15 days of discharge. "
            "You may appeal within 60 days of the rejection notice."
        ),
        expected_claims=[
            "The claim was rejected.",
            "The submission deadline is 15 days after discharge.",
            "An appeal may be filed within 60 days of the rejection notice.",
        ],
        expected_verdicts=["SUPPORTED", "CONTRADICTED", "SUPPORTED"],
        expected_action="REGENERATE",
        defect="intrinsic: numeric contradiction, 30 days rendered as 15",
        notes="The canonical intrinsic hallucination. Evidence exists and refutes the claim.",
    ),
    Fixture(
        fixture_id="F02",
        language="en",
        query="Why was my health insurance claim rejected?",
        context=[
            "Claims are rejected if submitted more than 30 days after the discharge date.",
        ],
        answer=(
            "Your claim was rejected because it was submitted more than 30 days after discharge. "
            "Additionally, pre-existing conditions are not covered under your policy."
        ),
        expected_claims=[
            "The claim was rejected due to submission more than 30 days after discharge.",
            "Pre-existing conditions are not covered under the policy.",
        ],
        expected_verdicts=["SUPPORTED", "UNVERIFIABLE"],
        expected_action="PRUNE",
        defect="extrinsic: pre-existing-condition clause absent from context",
        notes=(
            "Distinguishes UNVERIFIABLE from CONTRADICTED. The statement may well be true "
            "in the world; it is simply not supported by the retrieved evidence."
        ),
    ),
    Fixture(
        fixture_id="F03",
        language="en",
        query="How long do I have to appeal a rejected claim?",
        context=["Rejected claims may be appealed within 60 days of the rejection notice."],
        answer="You may appeal a rejected claim within 60 days of the rejection notice.",
        expected_claims=["An appeal may be filed within 60 days of the rejection notice."],
        expected_verdicts=["SUPPORTED"],
        expected_action="ACCEPT",
        defect="none — negative control",
        notes="Guards against a detector that flags everything. Precision must not be free.",
    ),
    Fixture(
        fixture_id="F04",
        language="en",
        query="What is the dental cover limit under my policy?",
        context=[],
        answer=(
            "Your policy provides dental cover up to Rs. 25,000 per year, "
            "including two free check-ups."
        ),
        expected_claims=[
            "The policy provides dental cover up to Rs. 25,000 per year.",
            "The policy includes two free dental check-ups.",
        ],
        expected_verdicts=["UNVERIFIABLE", "UNVERIFIABLE"],
        expected_action="ABSTAIN",
        defect="unsupported confidence: retrieval returned nothing, model answered anyway",
        notes="The failure mode a system that cannot abstain will always produce.",
    ),
    Fixture(
        fixture_id="F05",
        language="en",
        query="Is maternity covered in the first year?",
        context=["Maternity benefits are not available during the first policy year."],
        answer="Yes, maternity benefits are available during the first policy year.",
        expected_claims=["Maternity benefits are available during the first policy year."],
        expected_verdicts=["CONTRADICTED"],
        expected_action="REGENERATE",
        defect="intrinsic: negation dropped",
        notes=(
            "Negation loss is the single most damaging error class, and the one that "
            "machine translation introduces most often."
        ),
    ),
    Fixture(
        fixture_id="F06",
        language="hi",
        query="मेरा क्लेम क्यों रिजेक्ट हुआ?",
        context=[
            "डिस्चार्ज की तारीख से 30 दिन के बाद जमा किए गए क्लेम रिजेक्ट कर दिए जाते हैं।",
        ],
        answer="रिजेक्ट हो गया क्योंकि 15 दिन के अंदर जमा नहीं किया गया।",
        expected_claims=[
            "क्लेम रिजेक्ट हो गया।",
            "क्लेम जमा करने की समय सीमा 15 दिन है।",
        ],
        expected_verdicts=["SUPPORTED", "CONTRADICTED"],
        expected_action="REGENERATE",
        defect="intrinsic contradiction in Hindi, with the subject elided (pro-drop)",
        notes=(
            "The answer never names what was rejected. A decomposer that does not restore "
            "the elided subject produces a claim that is unverifiable for the wrong reason."
        ),
    ),
    Fixture(
        fixture_id="F07",
        language="ta",
        query="எனது காப்பீட்டு உரிமைகோரல் ஏன் நிராகரிக்கப்பட்டது?",
        context=[
            "வெளியேற்றப்பட்ட தேதியிலிருந்து 30 நாட்களுக்குப் பிறகு "
            "சமர்ப்பிக்கப்பட்ட உரிமைகோரல்கள் நிராகரிக்கப்படும்.",
        ],
        answer=(
            "உங்கள் உரிமைகோரல் 15 நாட்களுக்குள் சமர்ப்பிக்கப்படாததால் நிராகரிக்கப்பட்டது."
        ),
        expected_claims=[
            "உரிமைகோரல் நிராகரிக்கப்பட்டது.",
            "சமர்ப்பிப்பு காலக்கெடு 15 நாட்கள்.",
        ],
        expected_verdicts=["SUPPORTED", "CONTRADICTED"],
        expected_action="REGENERATE",
        defect="intrinsic contradiction in Tamil, two propositions inside one agglutinated form",
        notes=(
            "'சமர்ப்பிக்கப்படாததால்' packs negation, nominalisation and causality into one "
            "orthographic word. Whitespace-based splitting cannot decompose this."
        ),
    ),
]

FIXTURES_BY_ID: dict[str, Fixture] = {f.fixture_id: f for f in FIXTURES}


def fixtures_for(language: str | None = None) -> list[Fixture]:
    return [f for f in FIXTURES if language is None or f.language == language]


# ──────────────────────────────────────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class StubProvider:
    """Deterministic offline provider implementing ``LLMProvider``.

    Matching is by exact ``fixture_id`` marker in the prompt when present, else by
    query text, else by context overlap. Unmatched prompts fall back to a
    deterministic echo so the pipeline never blocks during development.
    """

    name: str = "stub"
    default_model: str = "stub-v1"
    declared_capabilities: set[Capability] = field(
        default_factory=lambda: {
            Capability.SYSTEM_ROLE,
            Capability.LOGPROBS,
            Capability.SEED,
            Capability.MULTI_SAMPLE,
        }
    )
    strict: bool = False
    """Raise on an unmatched prompt instead of echoing. Use in tests."""

    timeout: float = 0.0
    max_retries: int = 0

    # ── LLMProvider ───────────────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        prompt = "\n".join(m.content for m in request.messages)
        fixture = self._match(prompt)

        if fixture is None:
            if self.strict:
                raise ProviderError(
                    f"StubProvider: no fixture matches prompt: {prompt[:160]!r}", provider=self.name
                )
            text = f"[stub] No fixture matched. Echoing query: {self._last_user(request)[:200]}"
        else:
            text = fixture.answer

        completions = [
            self._completion(text, request, variant=0),
        ]
        # Extra samples perturb deterministically, so self-consistency (signal S4)
        # has something non-degenerate to measure while staying reproducible.
        for i in range(1, request.n):
            completions.append(self._completion(self._perturb(text, i), request, variant=i))

        return GenerationResponse(
            completions=completions,
            model=request.model or self.default_model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=sum(len(c.text.split()) for c in completions),
            ),
            latency_ms=0.0,
        )

    def capabilities(self, model: str) -> set[Capability]:
        return set(self.declared_capabilities)

    def available_models(self) -> Iterable[str]:
        return ["stub-v1"]

    def health_check(self) -> bool:
        return True

    def close(self) -> None:  # parity with network providers
        return None

    # ── matching ──────────────────────────────────────────────────────────────

    def _match(self, prompt: str) -> Fixture | None:
        if (m := re.search(r"\bF\d{2}\b", prompt)) and (
            f := FIXTURES_BY_ID.get(m.group(0))
        ) is not None:
            return f

        normalised = _normalise(prompt)
        for f in FIXTURES:
            if _normalise(f.query) and _normalise(f.query) in normalised:
                return f

        # Context-overlap fallback: useful when the harness reformats the query.
        best, best_score = None, 0.0
        for f in FIXTURES:
            if not f.context:
                continue
            score = max(_overlap(_normalise(c), normalised) for c in f.context)
            if score > best_score:
                best, best_score = f, score
        return best if best_score >= 0.6 else None

    @staticmethod
    def _last_user(request: GenerationRequest) -> str:
        for m in reversed(request.messages):
            if m.role == "user":
                return m.content
        return ""

    @staticmethod
    def _perturb(text: str, variant: int) -> str:
        """Deterministic paraphrase-ish variation for self-consistency sampling."""
        sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text) if s]
        if len(sentences) > 1:
            i = variant % len(sentences)
            return " ".join(sentences[i:] + sentences[:i])
        return text

    def _completion(self, text: str, request: GenerationRequest, variant: int) -> Completion:
        logprobs = None
        if request.want_logprobs:
            # Reproducible pseudo-logprobs derived from the token itself, so S3
            # can be exercised end-to-end without a real model.
            logprobs = [
                TokenLogProb(token=tok, logprob=self._pseudo_logprob(tok, variant))
                for tok in text.split()[:256]
            ]
        return Completion(text=text, finish_reason="stop", token_logprobs=logprobs)

    @staticmethod
    def _pseudo_logprob(token: str, variant: int) -> float:
        h = hashlib.sha256(f"{token}:{variant}".encode()).digest()[0] / 255.0
        return -0.05 - 1.5 * (h**2)  # in [-1.55, -0.05], skewed toward confident

    def __repr__(self) -> str:
        return f"<StubProvider fixtures={len(FIXTURES)} strict={self.strict}>"


# ──────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\sऀ-ॿ஀-௿]", re.UNICODE)


def _normalise(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def _overlap(a: str, b: str) -> float:
    """Fraction of ``a``'s tokens present in ``b``."""
    ta = set(a.split())
    if not ta:
        return 0.0
    return len(ta & set(b.split())) / len(ta)


def expected_summary() -> str:
    """Human-readable fixture table — printed by ``scripts/check_env.py``."""
    rows = [f"{'ID':<5} {'LANG':<5} {'ACTION':<11} DEFECT", "-" * 78]
    rows += [
        f"{f.fixture_id:<5} {f.language:<5} {f.expected_action:<11} {f.defect}" for f in FIXTURES
    ]
    return "\n".join(rows)

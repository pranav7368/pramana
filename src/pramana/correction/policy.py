"""Correction policy — decide what to do about a flagged answer, and do it.

Stage 5, the last part of the contribution.

**The risk this module is built around.** Unguided self-correction can make
correct answers worse: a model told to "check your work" will sometimes revise a
right answer into a wrong one (arXiv:2406.02378). A correction stage that ignores
this can lower faithfulness while appearing to improve it.

Four guard rails follow, and they are the design, not decoration:

1. **Evidence-guided, never blanket.** Correction fires only on specifically
   flagged claims, with the contradicting evidence surfaced. The model is never
   asked to review the whole answer.
2. **Bounded.** Loop depth is capped and latency is budgeted, so cost is provably
   bounded rather than emergent.
3. **Rollback.** A revision is accepted only if the claim-verdict profile strictly
   improves. Otherwise the previous answer is restored.
4. **Measured.** The regression rate -- answers made worse -- is recorded and
   reported. Concealing it in a reliability project would contradict the work's
   own premise.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pramana.generation.base import GenerationRequest, LLMProvider, Message
from pramana.schemas import (
    Action,
    Band,
    DetectionResult,
    Draft,
    Language,
    RetrievalResult,
    Verdict,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Verdict profile — the ordering that makes rollback decidable
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VerdictProfile:
    """Summary of a detection result, ordered so two answers can be compared.

    "Better" must be defined precisely or rollback cannot work. The ordering is:
    fewer contradictions first, then fewer unverifiable claims, then more supported
    ones. Contradictions dominate because a claim that conflicts with the evidence
    is a worse failure than one the evidence merely fails to confirm.
    """

    supported: int
    contradicted: int
    unverifiable: int

    @classmethod
    def of(cls, detection: DetectionResult) -> VerdictProfile:
        return cls(
            supported=detection.count(Verdict.SUPPORTED),
            contradicted=detection.count(Verdict.CONTRADICTED),
            unverifiable=detection.count(Verdict.UNVERIFIABLE),
        )

    def is_better_than(self, other: VerdictProfile) -> bool:
        if self.contradicted != other.contradicted:
            return self.contradicted < other.contradicted
        if self.unverifiable != other.unverifiable:
            return self.unverifiable < other.unverifiable
        return self.supported > other.supported

    @property
    def total(self) -> int:
        return self.supported + self.contradicted + self.unverifiable

    def __str__(self) -> str:
        return f"{self.supported}S/{self.contradicted}C/{self.unverifiable}U"


# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CorrectionPolicy:
    max_iterations: int = 2
    latency_budget_ms: float = 12_000
    abstain_below: float = 0.30
    """Confidence floor. Below this after exhausting iterations, abstain rather
    than return something the system does not trust."""

    weak_retrieval_score: float = 0.15
    """Below this, poor retrieval is the more likely cause than poor generation,
    so RE_RETRIEVE is preferred over REGENERATE."""

    prune_max_ratio: float = 0.5
    """PRUNE is only sensible while a majority of claims survive; beyond that the
    answer is being rewritten, not trimmed."""

    enabled_actions: frozenset[Action] = frozenset(Action)

    preserve_supported: bool = True
    """A revision may not drop claims the evidence already supported.

    The verdict-profile ordering prefers fewer unverifiable claims, so without
    this guard a rewrite from 3 supported + 1 unverifiable to 1 supported counts
    as an improvement: the answer got "cleaner" by saying less. Live mode's
    whole-answer coverage check catches some of these; this rule catches them
    with any verifier."""

    def decide(
        self,
        detection: DetectionResult,
        confidence_score: float,
        band: Band,
        retrieval: RetrievalResult,
        iteration: int,
    ) -> Action:
        """Select the next action. Pure and total -- always returns an action."""
        if not detection.n_claims:
            return Action.ACCEPT

        profile = VerdictProfile.of(detection)
        exhausted = iteration >= self.max_iterations

        # Nothing flagged and confidence holds -> done.
        if profile.contradicted == 0 and profile.unverifiable == 0 and band != "LOW":
            return self._allow(Action.ACCEPT)

        # A contradiction with usable evidence is the MOST correctable state: the
        # evidence disagrees, which means the evidence contains the right answer.
        # This is checked before any abstention rule -- an earlier version abstained
        # whenever no claim was yet supported, which meant a single-claim answer
        # contradicting good evidence was thrown away instead of fixed. Caught by an
        # end-to-end test; the policy unit tests all happened to include a supported
        # claim alongside the contradiction.
        correctable = (
            profile.contradicted > 0
            and retrieval.top_score >= self.weak_retrieval_score
            and not exhausted
        )
        if correctable:
            return self._allow(Action.REGENERATE)

        if profile.unverifiable > 0 and retrieval.top_score < self.weak_retrieval_score and not exhausted:
            return self._allow(Action.RE_RETRIEVE)

        if confidence_score < self.abstain_below or profile.supported == 0:
            return self._allow(Action.ABSTAIN)

        if exhausted:
            # PRUNE is a correction too. No mutating action may exceed the cap.
            return self._allow(Action.ABSTAIN)

        # A minority unverifiable while the rest is grounded: trim rather than redo.
        if profile.unverifiable / profile.total <= self.prune_max_ratio:
            return self._allow(Action.PRUNE)

        return self._allow(Action.REGENERATE if profile.contradicted else Action.ABSTAIN)

    def _allow(self, action: Action) -> Action:
        """Fall back to a permitted action when one is disabled by ablation."""
        if action in self.enabled_actions:
            return action
        if Action.ABSTAIN in self.enabled_actions:
            return Action.ABSTAIN
        return Action.ACCEPT


# ──────────────────────────────────────────────────────────────────────────────
# Correction prompts
# ──────────────────────────────────────────────────────────────────────────────

_REGENERATE_SYSTEM = {
    "en": (
        "You are correcting a previous answer that contained unsupported claims.\n"
        "Rules:\n"
        "1. Use ONLY the context. Every statement must be traceable to it.\n"
        "2. The listed problems are errors -- fix each one against the context.\n"
        "3. Keep everything that was correct. Do not rewrite the whole answer.\n"
        "4. Copy numbers, dates and negations from the context exactly.\n"
        "5. Answer in English. Output only the corrected answer."
    ),
    "hi": (
        "आप एक पिछले उत्तर को सुधार रहे हैं जिसमें असमर्थित दावे थे।\n"
        "नियम:\n"
        "1. केवल संदर्भ का उपयोग करें। हर बात संदर्भ से सिद्ध होनी चाहिए।\n"
        "2. नीचे दी गई समस्याएँ गलतियाँ हैं — हर एक को संदर्भ के अनुसार ठीक करें।\n"
        "3. जो सही था उसे वैसा ही रखें। पूरा उत्तर दोबारा न लिखें।\n"
        "4. संख्याएँ, तिथियाँ और 'नहीं' संदर्भ से हूबहू लें।\n"
        "5. हिंदी में केवल सुधरा हुआ उत्तर दें।"
    ),
    "ta": (
        "ஆதாரமற்ற கூற்றுகள் இருந்த முந்தைய பதிலைத் திருத்துகிறீர்கள்.\n"
        "விதிகள்:\n"
        "1. சூழலை மட்டுமே பயன்படுத்தவும். ஒவ்வொரு கூற்றும் சூழலில் இருக்க வேண்டும்.\n"
        "2. கீழே உள்ள சிக்கல்கள் பிழைகள் — ஒவ்வொன்றையும் சூழலின்படி திருத்தவும்.\n"
        "3. சரியாக இருந்தவற்றை அப்படியே வைக்கவும். முழுப் பதிலையும் மீண்டும் எழுத வேண்டாம்.\n"
        "4. எண்கள், தேதிகள், எதிர்மறைச் சொற்களைச் சூழலிலிருந்து அப்படியே எடுக்கவும்.\n"
        "5. தமிழில் திருத்தப்பட்ட பதிலை மட்டும் தரவும்."
    ),
}

_PROBLEM_LABEL = {
    Verdict.CONTRADICTED: {
        "en": "CONTRADICTS THE CONTEXT",
        "hi": "संदर्भ के विरुद्ध है",
        "ta": "சூழலுக்கு முரணானது",
    },
    Verdict.UNVERIFIABLE: {
        "en": "NOT SUPPORTED BY THE CONTEXT",
        "hi": "संदर्भ में नहीं है",
        "ta": "சூழலில் இல்லை",
    },
}


def build_correction_messages(
    draft: Draft, detection: DetectionResult, retrieval: RetrievalResult, language: Language
) -> tuple[Message, ...]:
    """Prompt naming exactly which claims failed and why.

    Listing the specific problems, rather than asking for a general review, is what
    makes this evidence-guided correction instead of the self-critique that is known
    to degrade correct answers.
    """
    problems = [
        f"- {_PROBLEM_LABEL[v.verdict].get(language, _PROBLEM_LABEL[v.verdict]['en'])}: "
        f"{v.claim.text}"
        for v in detection.claim_verdicts
        if v.is_hallucination
    ]
    system = _REGENERATE_SYSTEM.get(language, _REGENERATE_SYSTEM["en"])
    user = (
        f"Context:\n{retrieval.context_text()}\n\n"
        f"Previous answer:\n{draft.text}\n\n"
        f"Problems found:\n" + "\n".join(problems) + "\n\nCorrected answer:"
    )
    return (Message("system", system), Message("user", user))


# ──────────────────────────────────────────────────────────────────────────────
# Executor
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CorrectionOutcome:
    text: str
    action: Action
    profile_before: VerdictProfile
    profile_after: VerdictProfile | None
    accepted: bool
    regressed: bool
    """The candidate was strictly worse, and was rolled back."""

    latency_ms: float = 0.0
    detection_after: DetectionResult | None = None


@dataclass(slots=True)
class CorrectionExecutor:
    """Applies a chosen action and enforces the rollback guarantee."""

    provider: LLMProvider
    policy: CorrectionPolicy = field(default_factory=CorrectionPolicy)
    model: str = ""
    max_tokens: int = 512
    strict: bool = False
    abstention_message: Callable[[Language], str] | None = None

    def apply(
        self,
        action: Action,
        draft: Draft,
        detection: DetectionResult,
        retrieval: RetrievalResult,
        reverify: Callable[[str], DetectionResult],
        language: Language,
    ) -> CorrectionOutcome:
        """Execute ``action``.

        ``reverify`` re-runs decomposition and verification on a candidate answer.
        Passed in rather than constructed here so the executor stays independent of
        the detection stack -- and so tests can drive rollback deterministically.
        """
        started = time.perf_counter()
        before = VerdictProfile.of(detection)

        if action is Action.ACCEPT:
            return self._outcome(draft.text, action, before, None, True, False, started)

        if action is Action.ABSTAIN:
            message = (
                self.abstention_message(language)
                if self.abstention_message
                else "The available documents do not contain enough information to answer this."
            )
            return self._outcome(message, action, before, None, True, False, started)

        if action is Action.PRUNE:
            pruned = prune_claims(draft.text, detection)
            if not pruned.strip():
                # Removing everything is an abstention with extra steps; say so.
                message = (
                    self.abstention_message(language)
                    if self.abstention_message
                    else "The available documents do not contain enough information to answer this."
                )
                return self._outcome(message, Action.ABSTAIN, before, None, True, False, started)
            return self._accept_if_improved(pruned, action, before, reverify, started)

        # REGENERATE and RE_RETRIEVE both produce a fresh answer; the difference is
        # in the evidence supplied, which the caller has already adjusted.
        try:
            revised = self.provider.generate(
                GenerationRequest(
                    messages=build_correction_messages(draft, detection, retrieval, language),
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
            ).text.strip()
        except Exception as exc:
            if self.strict:
                raise
            log.warning("correction generation failed (%s); keeping the original answer", type(exc).__name__)
            return self._outcome(draft.text, action, before, None, False, False, started)

        if not revised:
            return self._outcome(draft.text, action, before, None, False, False, started)

        return self._accept_if_improved(revised, action, before, reverify, started)

    def _accept_if_improved(
        self,
        candidate: str,
        action: Action,
        before: VerdictProfile,
        reverify: Callable[[str], DetectionResult],
        started: float,
    ) -> CorrectionOutcome:
        """The rollback guarantee: keep a revision only if it strictly improves.

        Without this the correction stage can lower faithfulness while appearing to
        improve it -- the documented risk this module is designed around.
        """
        from pramana.generation.drafting import DraftGenerator

        if DraftGenerator.is_abstention(candidate):
            return self._outcome(candidate, Action.ABSTAIN, before, None, True, False, started)

        detection_after = reverify(candidate)
        after = VerdictProfile.of(detection_after)

        # A decomposer returning no claims is not proof that a factual answer was fixed.
        accepted = after.total > 0 and after.is_better_than(before)
        if accepted and self.policy.preserve_supported and after.supported < before.supported:
            log.info("rejecting %s: supported claims fell from %d to %d",
                     action.value, before.supported, after.supported)
            accepted = False
        regressed = before.is_better_than(after)
        outcome = self._outcome(candidate, action, before, after, accepted, regressed, started)
        outcome.detection_after = detection_after
        if accepted:
            return outcome

        log.info("rolling back %s: profile %s did not improve on %s", action.value, after, before)
        return outcome

    def _outcome(self, text, action, before, after, accepted, regressed, started) -> CorrectionOutcome:
        return CorrectionOutcome(
            text=text,
            action=action,
            profile_before=before,
            profile_after=after,
            accepted=accepted,
            regressed=regressed,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


# ──────────────────────────────────────────────────────────────────────────────
# PRUNE
# ──────────────────────────────────────────────────────────────────────────────


def prune_claims(text: str, detection: DetectionResult) -> str:
    """Remove the spans of unsupported claims, keeping the rest intact.

    Only span-linked claims can be excised. A claim rewritten during decomposition
    (to restore an elided Hindi subject, for instance) has no span, and guessing
    one would delete the wrong text -- so those are left in place and the verdict
    is carried forward instead.
    """
    spans = sorted(
        (
            v.claim.source_span
            for v in detection.claim_verdicts
            if v.is_hallucination and v.claim.source_span is not None
        ),
        key=lambda s: -s[0],
    )
    if not spans:
        return text

    out = text
    for start, end in spans:  # right to left, so earlier offsets stay valid
        if 0 <= start < end <= len(out):
            out = out[:start] + out[end:]

    return _tidy(out)


def _tidy(text: str) -> str:
    import re

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?।])", r"\1", text)
    text = re.sub(r"([.।])\s*\1+", r"\1", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RegressionTracker:
    """Counts answers made worse by correction.

    Reported alongside the improvement figure. A correction stage that improves
    60% of answers and degrades 20% is a very different system from one that
    improves 45% and degrades none, and the headline number alone cannot tell them
    apart (`03_PROPOSAL.md` §4.4).
    """

    attempts: int = 0
    improved: int = 0
    regressed: int = 0
    unchanged: int = 0

    def record(self, outcome: CorrectionOutcome) -> None:
        if outcome.action in (Action.ACCEPT, Action.ABSTAIN):
            return
        self.attempts += 1
        if outcome.regressed:
            self.regressed += 1
        elif outcome.accepted:
            self.improved += 1
        else:
            self.unchanged += 1

    @property
    def regression_rate(self) -> float:
        return self.regressed / self.attempts if self.attempts else 0.0

    @property
    def improvement_rate(self) -> float:
        return self.improved / self.attempts if self.attempts else 0.0

    def summary(self) -> str:
        if not self.attempts:
            return "no corrections attempted"
        return (
            f"{self.attempts} corrections · {self.improved} improved "
            f"({self.improvement_rate:.0%}) · {self.regressed} regressed and rolled back "
            f"({self.regression_rate:.0%}) · {self.unchanged} unchanged"
        )

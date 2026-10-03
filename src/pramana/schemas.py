"""Shared data contracts for the whole pipeline.

Every stage boundary is one of these types. Keeping them in a single module with
no heavy imports means each stage can be tested in isolation against fixtures,
without loading a model or touching the network -- which is what makes this
project developable on a laptop with no GPU.

Corresponds to docs/04_SYSTEM_ARCHITECTURE.md §3.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

Language = Literal["en", "hi", "ta"]
LANGUAGES: tuple[Language, ...] = ("en", "hi", "ta")

Script = Literal["native", "roman", "mixed"]


class Verdict(StrEnum):
    """Outcome of verifying one claim against the retrieved evidence.

    CONTRADICTED and UNVERIFIABLE are deliberately distinct. Most prior work
    collapses them into a single "hallucination" label, but they demand different
    corrective actions: a contradicted claim should be *corrected*, an
    unverifiable one *removed* or abstained on.
    """

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class Action(StrEnum):
    """Correction actions. See docs/04_SYSTEM_ARCHITECTURE.md §4.6."""

    ACCEPT = "ACCEPT"
    REGENERATE = "REGENERATE"
    RE_RETRIEVE = "RE_RETRIEVE"
    PRUNE = "PRUNE"
    ABSTAIN = "ABSTAIN"


# ──────────────────────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    language: Language
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"chunk {self.chunk_id} has no text")


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    rank: int
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """The score the pipeline should reason about -- rerank wins when present."""
        return self.rerank_score if self.rerank_score is not None else self.fused_score


@dataclass(slots=True)
class RetrievalResult:
    query: str
    normalized_query: str
    language: Language
    script: Script
    chunks: list[RetrievedChunk] = field(default_factory=list)

    # ── confidence signal S1 ──────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """No evidence retrieved. The generator must abstain rather than answer."""
        return not self.chunks

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    @property
    def score_margin(self) -> float:
        """Gap between the best and second-best hit.

        A large margin means one passage is clearly the right one; a flat
        distribution means retrieval could not discriminate, which is a strong
        predictor of an ungrounded answer.
        """
        if len(self.chunks) < 2:
            return 0.0
        return self.chunks[0].score - self.chunks[1].score

    @property
    def mean_top_k_score(self) -> float:
        return sum(c.score for c in self.chunks) / len(self.chunks) if self.chunks else 0.0

    def signals(self) -> dict[str, float]:
        return {
            "s1_top_score": self.top_score,
            "s1_score_margin": self.score_margin,
            "s1_mean_score": self.mean_top_k_score,
            "s1_n_chunks": float(len(self.chunks)),
        }

    def context_text(self, separator: str = "\n\n") -> str:
        return separator.join(c.chunk.text for c in self.chunks)

    def chunk_ids(self) -> list[str]:
        return [c.chunk.chunk_id for c in self.chunks]


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Draft:
    """A generated answer plus everything needed to judge its uncertainty."""

    text: str
    language: Language
    samples: list[str] = field(default_factory=list)
    """Stochastic resamples for signal S4. MUST be generated at temperature > 0 --
    at temperature 0 they are identical and the signal collapses to a constant."""

    mean_token_logprob: float | None = None
    """Signal S3. ``None`` when the provider returns no logprobs -- which is the
    case for every free endpoint measured so far. The fusion model is refitted
    without S3 rather than imputing a value, because imputation would corrupt
    calibration silently."""

    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    sample_temperature: float = 0.0

    @property
    def has_logprobs(self) -> bool:
        return self.mean_token_logprob is not None

    SHORT_ANSWER_TOKENS = 4
    """Below this length, self-consistency is uninformative -- see ``s4_reliable``."""

    @property
    def self_consistency(self) -> float:
        """Signal S4: agreement across resamples, in [0, 1].

        Token-level Jaccard averaged over all sample pairs. Cheap, language-neutral,
        and needs no extra model -- important because it is currently the *only*
        available uncertainty signal.

        Returns 0.0 when there are fewer than two samples: with nothing to compare,
        the honest answer is "no evidence of consistency", not "perfectly consistent".
        """
        pool = [self.text, *self.samples]
        if len(pool) < 2:
            return 0.0

        token_sets = [set(s.split()) for s in pool]
        scores: list[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                a, b = token_sets[i], token_sets[j]
                union = a | b
                scores.append(len(a & b) / len(union) if union else 1.0)
        return sum(scores) / len(scores)

    @property
    def s4_reliable(self) -> bool:
        """Whether self-consistency carries usable information for this answer.

        Observed live on 2026-08-10: a Tamil yes/no question produced the answer
        "இல்லை" ("No") and three identical samples **at temperature 0.9**. That is
        not a sampling bug -- a one-token answer is stable under any temperature,
        so S4 is pinned at 1.0 and says nothing about whether the model was right.

        This matters because S3 is unavailable on every free endpoint measured, so
        S4 is the only uncertainty signal available. For short answers the
        confidence estimate must rest on S1 (retrieval quality) and S2 (entailment
        strength) instead. The fusion model receives this as an explicit feature
        rather than being left to infer it from answer length.
        """
        return len(self.text.split()) >= self.SHORT_ANSWER_TOKENS and len(self.samples) >= 2

    @property
    def sample_dispersion(self) -> float:
        """1 - self_consistency. The S3 substitute when logprobs are unavailable."""
        return 1.0 - self.self_consistency

    def signals(self) -> dict[str, float]:
        out = {
            "s4_self_consistency": self.self_consistency,
            "s4_dispersion": self.sample_dispersion,
            "s4_n_samples": float(len(self.samples)),
            # Lets the fusion model learn to discount S4 on short answers rather
            # than treating a trivially-stable 1.0 as evidence of correctness.
            "s4_reliable": float(self.s4_reliable),
            "answer_tokens": float(len(self.text.split())),
        }
        if self.mean_token_logprob is not None:
            out["s3_mean_logprob"] = self.mean_token_logprob
            out["s3_perplexity"] = math.exp(-self.mean_token_logprob)
        return out

    def degenerate_samples(self) -> bool:
        """True when every sample is identical -- signal S4 carries no information.

        Almost always means sampling ran at temperature 0. Callers should warn
        rather than silently fitting a confidence model on a constant feature.
        """
        return bool(self.samples) and len({self.text, *self.samples}) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Claim:
    """An atomic, self-contained, independently verifiable proposition."""

    claim_id: str
    text: str
    language: Language
    source_span: tuple[int, int] | None = None
    """Character offsets into the draft, so PRUNE can excise exactly this claim.
    ``None`` when the claim was rewritten during decomposition (e.g. an elided
    Hindi subject was restored) and no longer maps to a contiguous span."""


@dataclass(slots=True)
class ClaimVerdict:
    claim: Claim
    verdict: Verdict
    entailment_prob: float = 0.0
    contradiction_prob: float = 0.0
    neutral_prob: float = 0.0
    supporting_chunk_ids: list[str] = field(default_factory=list)
    verification_language: Language = "en"
    arm: Literal["native", "translate", "hybrid"] = "native"

    @property
    def margin(self) -> float:
        """Gap between the top and second class probability.

        Low margin means the verifier was near-indifferent, which matters more
        than which label won -- it feeds signal S2 as a confidence weight.
        """
        probs = sorted(
            (self.entailment_prob, self.contradiction_prob, self.neutral_prob), reverse=True
        )
        return probs[0] - probs[1]

    @property
    def is_hallucination(self) -> bool:
        return self.verdict is not Verdict.SUPPORTED


@dataclass(slots=True)
class DetectionResult:
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    partial: bool = False
    """Every claim is supported, but the evidence covers only part of the
    question. The answer may be released, labelled as partial."""

    @property
    def n_claims(self) -> int:
        return len(self.claim_verdicts)

    def count(self, verdict: Verdict) -> int:
        return sum(1 for v in self.claim_verdicts if v.verdict is verdict)

    @property
    def supported_ratio(self) -> float:
        return self.count(Verdict.SUPPORTED) / self.n_claims if self.n_claims else 0.0

    @property
    def has_contradiction(self) -> bool:
        return self.count(Verdict.CONTRADICTED) > 0

    @property
    def hallucination_rate(self) -> float:
        if not self.n_claims:
            return 0.0
        return sum(1 for v in self.claim_verdicts if v.is_hallucination) / self.n_claims

    def signals(self) -> dict[str, float]:
        n = self.n_claims or 1
        margins = [v.margin for v in self.claim_verdicts]
        return {
            "s2_supported_ratio": self.supported_ratio,
            "s2_contradicted_ratio": self.count(Verdict.CONTRADICTED) / n,
            "s2_unverifiable_ratio": self.count(Verdict.UNVERIFIABLE) / n,
            "s2_mean_margin": sum(margins) / len(margins) if margins else 0.0,
            "s2_min_margin": min(margins) if margins else 0.0,
            "s2_n_claims": float(self.n_claims),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Confidence & assurance
# ──────────────────────────────────────────────────────────────────────────────


Band = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass(slots=True)
class ConfidenceReport:
    score: float
    raw_score: float
    band: Band
    signals: dict[str, float] = field(default_factory=dict)
    calibrator: str = "none"
    missing_signals: list[str] = field(default_factory=list)
    """Signals unavailable for this run -- e.g. S3 when the provider returns no
    logprobs. Recorded so results can state honestly what the model was fitted on."""


@dataclass(slots=True)
class AssuranceResult:
    """The pipeline's output: an answer that knows how much to trust itself."""

    final_answer: str
    language: Language
    confidence: ConfidenceReport
    detection: DetectionResult
    action_history: list[Action] = field(default_factory=list)
    iterations: int = 0
    regressed: bool = False
    """At least one candidate was strictly worse and was rolled back."""

    abstained: bool = False
    evidence_chunk_ids: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    trace_id: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    correction_attempts: int = 0
    correction_regressions: int = 0
    rolled_back: bool = False
    stop_reason: str = ""
    partial: bool = False
    """Released answer covers only part of the question; the rest is not in the evidence."""

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_ms.values())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

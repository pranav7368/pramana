"""Human-verified LLM pre-annotation.

Cuts annotation from ~45 hours to ~7-15 while staying methodologically sound. The
whole design turns on one distinction:

* **Circular (invalid):** the system labels its own output, then is scored against
  those labels. F1 is 1.0 by construction. Worse here than in most projects,
  because the thesis *is* that LLM verification is unreliable in Indic languages —
  using an LLM to build the ground truth assumes the conclusion.
* **Sound:** a **different** model proposes labels, a human verifies a stratified
  sample, and **agreement between them is reported**.

**The agreement number is itself a finding.** "LLM-as-judge agrees with human
annotators at κ = 0.82 in English but κ = 0.51 in Tamil" directly supports the
project's argument and gives the judge baseline a
reliability figure rather than only an F1.

Two properties are enforced in code rather than left to discipline:

1. The judge provider must differ from the system provider (``JudgeConflictError``).
2. Sampling is stratified by (language, proposed label), because CONTRADICTED is
   the rare class and the one most worth validating — uniform sampling would
   barely touch it.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pramana.generation.base import GenerationRequest, LLMProvider, Message
from pramana.schemas import Language, Verdict

log = logging.getLogger(__name__)

Source = Literal["llm", "human", "adjudicated"]

LABELS: tuple[str, ...] = (Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNVERIFIABLE)


class JudgeConflictError(RuntimeError):
    """The annotator and the system under test share a provider or model.

    Refused rather than warned: a run that produces circular labels looks exactly
    like a valid one, and the resulting numbers would be indefensible in review.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Records
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AnnotationItem:
    """One claim awaiting a label, with the evidence it must be judged against."""

    item_id: str
    qid: str
    language: Language
    question: str
    claim: str
    evidence: list[str]
    system: str = ""

    @property
    def evidence_text(self) -> str:
        return "\n\n".join(self.evidence)


@dataclass(slots=True)
class Annotation:
    item_id: str
    label: str
    source: Source
    annotator: str = ""
    confidence: float | None = None
    note: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"invalid label {self.label!r}; expected one of {LABELS}")


# ──────────────────────────────────────────────────────────────────────────────
# Agreement
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AgreementReport:
    kappa: float
    raw_agreement: float
    n: int
    confusion: dict[tuple[str, str], int]
    per_label_agreement: dict[str, float]

    @property
    def interpretation(self) -> str:
        """Landis & Koch bands. The project targets κ ≥ 0.70."""
        k = self.kappa
        if k < 0.0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"

    @property
    def acceptable(self) -> bool:
        """Below 0.60 the guidelines are revised and the batch redone
        (see docs/13_VALIDATION_AND_HUMAN_REVIEW.md)."""
        return self.kappa >= 0.60

    def __str__(self) -> str:
        flag = "" if self.acceptable else "   ← BELOW THRESHOLD, revise guidelines"
        return (
            f"κ={self.kappa:.3f} ({self.interpretation}) · "
            f"raw agreement {self.raw_agreement:.1%} · n={self.n}{flag}"
        )


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> AgreementReport:
    """Cohen's κ between two label sequences.

    κ rather than raw agreement because the label distribution is skewed: SUPPORTED
    will dominate, so two raters who both always guess SUPPORTED would show ~85%
    raw agreement while agreeing on nothing informative. κ corrects for chance.
    """
    if len(a) != len(b):
        raise ValueError(f"{len(a)} labels vs {len(b)} — must be paired")
    n = len(a)
    if n == 0:
        return AgreementReport(0.0, 0.0, 0, {}, {})

    confusion: dict[tuple[str, str], int] = {}
    for x, y in zip(a, b, strict=True):
        confusion[(x, y)] = confusion.get((x, y), 0) + 1

    observed = sum(c for (x, y), c in confusion.items() if x == y) / n

    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        (count_a.get(label, 0) / n) * (count_b.get(label, 0) / n) for label in LABELS
    )

    kappa = (observed - expected) / (1 - expected) if expected < 1.0 else 1.0

    per_label: dict[str, float] = {}
    for label in LABELS:
        relevant = [(x, y) for x, y in zip(a, b, strict=True) if x == label or y == label]
        if relevant:
            per_label[label] = sum(1 for x, y in relevant if x == y) / len(relevant)

    return AgreementReport(
        kappa=kappa,
        raw_agreement=observed,
        n=n,
        confusion=confusion,
        per_label_agreement=per_label,
    )


def agreement_by_language(
    pairs: Sequence[tuple[Language, str, str]],
) -> dict[Language, AgreementReport]:
    """Per-language agreement. **This is the finding**, not a diagnostic.

    A judge that agrees with humans in English but not in Tamil is evidence for the
    project's central claim, so it must never be collapsed into a single number.
    """
    grouped: dict[Language, tuple[list[str], list[str]]] = {}
    for lang, x, y in pairs:
        a, b = grouped.setdefault(lang, ([], []))
        a.append(x)
        b.append(y)
    return {lang: cohen_kappa(a, b) for lang, (a, b) in grouped.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────


def stratified_sample(
    items: Sequence[AnnotationItem],
    proposed: dict[str, str],
    *,
    fraction: float = 0.30,
    min_per_stratum: int = 5,
    seed: int = 42,
) -> list[AnnotationItem]:
    """Select items for human review, stratified by (language, proposed label).

    Uniform random sampling would be wrong here. CONTRADICTED is the rare class and
    the one whose validation matters most — a uniform 30% draw might contain almost
    none of it, leaving the judge's behaviour on the critical class unmeasured.
    Each stratum therefore contributes at least ``min_per_stratum`` items where it
    has them.
    """
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[AnnotationItem]] = {}
    for item in items:
        key = (item.language, proposed.get(item.item_id, "UNKNOWN"))
        strata.setdefault(key, []).append(item)

    selected: list[AnnotationItem] = []
    for key, members in sorted(strata.items()):
        take = max(min_per_stratum, round(len(members) * fraction))
        take = min(take, len(members))
        pool = list(members)
        rng.shuffle(pool)
        selected.extend(pool[:take])
        log.debug("stratum %s: %d of %d", key, take, len(members))

    rng.shuffle(selected)  # randomise review order so fatigue does not track label
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# LLM annotator
# ──────────────────────────────────────────────────────────────────────────────

_ANNOTATOR_SYSTEM = """You are annotating claims for a hallucination-detection dataset.

For each CLAIM, decide its relationship to the EVIDENCE:

  SUPPORTED     - the evidence states or directly entails the claim
  CONTRADICTED  - the evidence states something incompatible with the claim
  UNVERIFIABLE  - the evidence neither supports nor refutes the claim

CRITICAL RULE: judge against the EVIDENCE ONLY, never against your own knowledge
of the world. A claim that is true in reality but absent from the evidence is
UNVERIFIABLE, not SUPPORTED. This is the single most common annotation error.

Pay close attention to numbers, dates and negation. A different number, or a
dropped or added "not", is CONTRADICTED — not UNVERIFIABLE.

Reply with one word only: SUPPORTED, CONTRADICTED, or UNVERIFIABLE."""

_LABEL_RE = re.compile(r"\b(SUPPORTED|CONTRADICTED|UNVERIFIABLE)\b", re.I)


@dataclass(slots=True)
class LLMAnnotator:
    """Proposes labels with a model deliberately different from the system's.

    Args:
        provider: the judge. Must not be the provider under test.
        model: judge model id.
        system_provider: name of the provider the *system* uses, for the conflict
            check. Pass it — the check is the point.
    """

    provider: LLMProvider
    model: str = ""
    system_provider: str = ""
    system_model: str = ""
    annotator_id: str = "llm"
    judge_provider: str = ""
    """Underlying provider name.

    Required when ``provider`` is an ``LLMRouter``: the router reports its own name
    ("router"), not the backend it dispatches to, so the conflict check silently
    passed against it. The guard is the point of this class, so the caller must
    state which provider is actually judging.
    """

    def __post_init__(self) -> None:
        judge = self.judge_provider or self._resolve_provider_name()
        if self.system_provider and judge and judge == self.system_provider:
            raise JudgeConflictError(
                f"annotator and system both use provider {judge!r}. "
                "Labelling a system's output with the same model is circular: the "
                "resulting F1 measures self-consistency, not correctness. "
                "Configure a second provider (e.g. system=groq, judge=google)."
            )
        if self.system_model and self.model and self.model == self.system_model:
            raise JudgeConflictError(
                f"annotator and system both use model {self.model!r}. See above."
            )

    def _resolve_provider_name(self) -> str:
        """Unwrap a router to the backend it will actually dispatch to.

        A router bound to exactly one provider is that provider. Bound to several,
        it could dispatch to any of them — including the system's — so the caller
        must name the judge explicitly rather than let the check guess.
        """
        bound = getattr(self.provider, "_bound", None)
        if bound is not None:
            names = [b.spec.name for b in bound]
            if len(names) == 1:
                return names[0]
            if self.system_provider and self.system_provider in names:
                raise JudgeConflictError(
                    f"the judge router may dispatch to {names}, which includes the "
                    f"system's provider {self.system_provider!r}. Bind the judge to a "
                    "single provider, or pass judge_provider= explicitly."
                )
            return ""
        return getattr(self.provider, "name", "")

    def annotate(self, item: AnnotationItem) -> Annotation:
        if not item.evidence:
            # No evidence retrieved: the verdict is UNVERIFIABLE by definition, so
            # spending a judge call would only add noise.
            return Annotation(
                item_id=item.item_id,
                label=Verdict.UNVERIFIABLE.value,
                source="llm",
                annotator=self.annotator_id,
                confidence=1.0,
                note="no evidence retrieved — unverifiable by definition",
            )

        prompt = (
            f"EVIDENCE:\n{item.evidence_text}\n\n"
            f"QUESTION: {item.question}\n\n"
            f"CLAIM: {item.claim}"
        )
        try:
            out = self.provider.generate(
                GenerationRequest(
                    messages=(Message("system", _ANNOTATOR_SYSTEM), Message("user", prompt)),
                    model=self.model,
                    temperature=0.0,
                    max_tokens=12,
                )
            ).text
        except Exception as exc:
            log.warning("annotator failed on %s (%s); marking for human review", item.item_id, exc)
            return Annotation(
                item_id=item.item_id,
                label=Verdict.UNVERIFIABLE.value,
                source="llm",
                annotator=self.annotator_id,
                confidence=0.0,
                note=f"judge error: {exc}",
            )

        match = _LABEL_RE.search(out)
        if match is None:
            # An unreadable judgement must not become a confident label.
            return Annotation(
                item_id=item.item_id,
                label=Verdict.UNVERIFIABLE.value,
                source="llm",
                annotator=self.annotator_id,
                confidence=0.0,
                note=f"unparseable judge output: {out[:60]!r}",
            )

        return Annotation(
            item_id=item.item_id,
            label=match.group(1).upper(),
            source="llm",
            annotator=self.annotator_id,
            confidence=0.9,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AnnotationStore:
    """Append-only JSONL store, resumable across sessions.

    Annotation runs to many hours across several sittings; a store that loses work
    on interruption would not survive contact with the actual task. Later entries
    for the same item override earlier ones, so a human label supersedes the LLM's.
    """

    path: Path
    items: dict[str, AnnotationItem] = field(default_factory=dict)
    annotations: dict[str, list[Annotation]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except ValueError:
                log.warning("skipping unreadable annotation line")
                continue
            if d.get("_type") == "item":
                d.pop("_type")
                self.items[d["item_id"]] = AnnotationItem(**d)
            elif d.get("_type") == "annotation":
                d.pop("_type")
                self.annotations.setdefault(d["item_id"], []).append(Annotation(**d))

    def _append(self, kind: str, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"_type": kind, **payload}, ensure_ascii=False) + "\n")
            fh.flush()  # survives an interrupted session

    def add_item(self, item: AnnotationItem) -> None:
        if item.item_id not in self.items:
            self.items[item.item_id] = item
            self._append("item", asdict(item))

    def add_annotation(self, annotation: Annotation) -> None:
        self.annotations.setdefault(annotation.item_id, []).append(annotation)
        self._append("annotation", asdict(annotation))

    # ── queries ───────────────────────────────────────────────────────────────

    def label(self, item_id: str, source: Source | None = None) -> str | None:
        """Most recent label, optionally restricted to one source."""
        entries = self.annotations.get(item_id, [])
        if source:
            entries = [a for a in entries if a.source == source]
        return entries[-1].label if entries else None

    def final_label(self, item_id: str) -> str | None:
        """Human or adjudicated label if one exists, else the LLM's.

        The precedence that makes the dataset trustworthy: wherever a human looked,
        the human decides.
        """
        for source in ("adjudicated", "human"):
            if (label := self.label(item_id, source)) is not None:  # type: ignore[arg-type]
                return label
        return self.label(item_id, "llm")

    def needs_human(self) -> list[AnnotationItem]:
        return [i for i in self.items.values() if self.label(i.item_id, "human") is None]

    def disagreements(self) -> list[tuple[AnnotationItem, str, str]]:
        """Items where the human overrode the LLM. The most informative subset:
        these are where the judge is unreliable, and reading them tells you *how*."""
        out = []
        for item_id, item in self.items.items():
            llm = self.label(item_id, "llm")
            human = self.label(item_id, "human")
            if llm and human and llm != human:
                out.append((item, llm, human))
        return out

    def agreement(self) -> dict[Language, AgreementReport]:
        pairs = [
            (item.language, llm, human)
            for item_id, item in self.items.items()
            if (llm := self.label(item_id, "llm")) and (human := self.label(item_id, "human"))
        ]
        return agreement_by_language(pairs)

    def summary(self) -> str:
        reviewed = sum(1 for i in self.items if self.label(i, "human"))
        proposed = sum(1 for i in self.items if self.label(i, "llm"))
        return (
            f"{len(self.items)} items · {proposed} LLM-labelled · "
            f"{reviewed} human-reviewed ({reviewed / len(self.items):.0%})"
            if self.items
            else "empty store"
        )

    def export(self, path: Path) -> int:
        """Write final labels for the evaluation harness.

        ``label_source`` is exported per item so results can be reported separately
        for human-verified and LLM-only labels — required wherever no human reader
        was available (Tamil, risk R2).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with path.open("w", encoding="utf-8") as fh:
            for item_id, item in sorted(self.items.items()):
                label = self.final_label(item_id)
                if label is None:
                    continue
                human = self.label(item_id, "human")
                fh.write(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "qid": item.qid,
                            "language": item.language,
                            "claim": item.claim,
                            "label": label,
                            "label_source": "human" if human else "llm",
                            "llm_label": self.label(item_id, "llm"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
        return written

"""Evaluation metrics — the numbers that go in the report.

Written dependency-free so
the harness runs in the base install on the dev laptop.

Two properties are enforced by construction rather than left to discipline:

* **Nothing is reported without an uncertainty interval.** ``bootstrap_ci`` is
  built into the result types. With a few hundred questions per language the
  interval is often wide enough to change the conclusion, and hiding that would be
  the most likely way this project reports something false.
* **Nothing is averaged across languages.** Metrics are computed per language and
  the aggregate is presented alongside, never instead. A detector that works in
  English and fails in Tamil must not be able to look adequate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pramana.schemas import LANGUAGES, Language, Verdict

# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int

    def __str__(self) -> str:
        return f"{self.label:<14} P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} n={self.support}"


@dataclass(slots=True)
class ClassificationReport:
    per_class: dict[str, ClassMetrics]
    confusion: dict[tuple[str, str], int]
    """(gold, predicted) -> count. The CONTRADICTED/UNVERIFIABLE cell matters most:
    confusing the two sends the wrong correction action."""

    n: int

    @property
    def macro_f1(self) -> float:
        """The headline number.

        Macro rather than accuracy because SUPPORTED will dominate the label
        distribution -- accuracy would reward a detector that flags nothing.
        """
        if not self.per_class:
            return 0.0
        return sum(m.f1 for m in self.per_class.values()) / len(self.per_class)

    @property
    def accuracy(self) -> float:
        correct = sum(c for (g, p), c in self.confusion.items() if g == p)
        return correct / self.n if self.n else 0.0

    def binary_f1(self, positive: str = Verdict.SUPPORTED.value) -> float:
        """SUPPORTED vs. not -- the hallucination-detection number."""
        tp = sum(c for (g, p), c in self.confusion.items() if g == positive and p == positive)
        fp = sum(c for (g, p), c in self.confusion.items() if g != positive and p == positive)
        fn = sum(c for (g, p), c in self.confusion.items() if g == positive and p != positive)
        return _f1(tp, fp, fn)

    def format_confusion(self) -> str:
        labels = sorted({lab for pair in self.confusion for lab in pair})
        head = "gold \\ pred".ljust(16) + "".join(lab[:5].rjust(8) for lab in labels)
        rows = [
            lab.ljust(16) + "".join(str(self.confusion.get((lab, p), 0)).rjust(8) for p in labels)
            for lab in labels
        ]
        return "\n".join([head, *rows])


def classification_report(gold: Sequence[str], pred: Sequence[str]) -> ClassificationReport:
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")

    confusion: dict[tuple[str, str], int] = {}
    for g, p in zip(gold, pred, strict=True):
        confusion[(g, p)] = confusion.get((g, p), 0) + 1

    per_class: dict[str, ClassMetrics] = {}
    for label in sorted(set(gold) | set(pred)):
        tp = confusion.get((label, label), 0)
        fp = sum(c for (g, p), c in confusion.items() if g != label and p == label)
        fn = sum(c for (g, p), c in confusion.items() if g == label and p != label)
        support = sum(c for (g, _), c in confusion.items() if g == label)
        per_class[label] = ClassMetrics(
            label=label,
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else 0.0,
            f1=_f1(tp, fp, fn),
            support=support,
        )

    return ClassificationReport(per_class=per_class, confusion=confusion, n=len(gold))


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RetrievalMetrics:
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    empty_rate: float
    k: int
    n: int


def retrieval_metrics(
    retrieved: Sequence[Sequence[str]], gold: Sequence[Sequence[str]], *, k: int = 5
) -> RetrievalMetrics:
    """Recall@k, nDCG@k, MRR and empty rate over a set of queries.

    Retrieval bounds everything downstream: an answer cannot be grounded in
    evidence that was never retrieved, so a generation failure must not be
    attributed to the model without checking this first.
    """
    if len(retrieved) != len(gold):
        raise ValueError(f"{len(retrieved)} result lists but {len(gold)} gold lists")
    if not retrieved:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, k, 0)

    recalls, ndcgs, rrs, empties = [], [], [], 0
    for got, want in zip(retrieved, gold, strict=True):
        top = list(got[:k])
        if not got:
            empties += 1
        want_set = set(want)
        if not want_set:
            continue

        recalls.append(len(want_set & set(top)) / len(want_set))

        dcg = sum(1 / math.log2(i + 2) for i, cid in enumerate(top) if cid in want_set)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(want_set), k)))
        ndcgs.append(dcg / ideal if ideal else 0.0)

        rrs.append(
            next((1 / (i + 1) for i, cid in enumerate(top) if cid in want_set), 0.0)
        )

    return RetrievalMetrics(
        recall_at_k=_mean(recalls),
        ndcg_at_k=_mean(ndcgs),
        mrr=_mean(rrs),
        empty_rate=empties / len(retrieved),
        k=k,
        n=len(retrieved),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Abstention — promoted to first-class after finding P1
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AbstentionMetrics:
    """Hallucination rate and abstention rate must be read together.

    A system that never answers has a hallucination rate of zero and is useless.
    Finding P1 showed abstention behaviour differing sharply by language under
    identical prompts, so this cannot be an afterthought.
    """

    rate: float
    precision: float
    recall: float
    over_abstention_rate: float
    """Answerable questions the system refused. The failure mode a
    hallucination-only view is blind to."""

    n: int

    def __str__(self) -> str:
        return (
            f"abstained {self.rate:.1%} · precision {self.precision:.3f} · "
            f"recall {self.recall:.3f} · over-abstention {self.over_abstention_rate:.1%}"
        )


def abstention_metrics(
    abstained: Sequence[bool], should_abstain: Sequence[bool]
) -> AbstentionMetrics:
    if len(abstained) != len(should_abstain):
        raise ValueError(f"{len(abstained)} decisions but {len(should_abstain)} gold flags")
    if not abstained:
        return AbstentionMetrics(0.0, 0.0, 0.0, 0.0, 0)

    tp = sum(1 for a, s in zip(abstained, should_abstain, strict=True) if a and s)
    fp = sum(1 for a, s in zip(abstained, should_abstain, strict=True) if a and not s)
    fn = sum(1 for a, s in zip(abstained, should_abstain, strict=True) if not a and s)
    answerable = sum(1 for s in should_abstain if not s)

    return AbstentionMetrics(
        rate=sum(abstained) / len(abstained),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        over_abstention_rate=fp / answerable if answerable else 0.0,
        n=len(abstained),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Correction
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CorrectionMetrics:
    hallucination_before: float
    hallucination_after: float
    relative_reduction: float
    regression_rate: float
    improvement_rate: float
    relevance_retention: float
    n_attempts: int

    @property
    def net_improvement(self) -> float:
        """The honest headline: improvements minus regressions."""
        return self.improvement_rate - self.regression_rate

    def __str__(self) -> str:
        return (
            f"hallucination {self.hallucination_before:.1%} -> {self.hallucination_after:.1%} "
            f"({self.relative_reduction:+.1%}) · improved {self.improvement_rate:.1%} · "
            f"REGRESSED {self.regression_rate:.1%} · net {self.net_improvement:+.1%}"
        )


def correction_metrics(
    before: Sequence[float],
    after: Sequence[float],
    regressed: Sequence[bool],
    relevance_before: Sequence[float] | None = None,
    relevance_after: Sequence[float] | None = None,
) -> CorrectionMetrics:
    """Correction effect, with regression reported at equal prominence.

    A stage that improves 60% and degrades 20% is a different system from one that
    improves 45% and degrades none; the improvement figure alone cannot tell them
    apart.
    """
    if not before:
        return CorrectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0)

    h_before, h_after = _mean(before), _mean(after)
    improved = sum(1 for b, a in zip(before, after, strict=True) if a < b)

    retention = 1.0
    if relevance_before and relevance_after:
        rb = _mean(relevance_before)
        retention = _mean(relevance_after) / rb if rb else 1.0

    return CorrectionMetrics(
        hallucination_before=h_before,
        hallucination_after=h_after,
        relative_reduction=(h_before - h_after) / h_before if h_before else 0.0,
        regression_rate=sum(regressed) / len(regressed) if regressed else 0.0,
        improvement_rate=improved / len(before),
        relevance_retention=retention,
        n_attempts=len(before),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Interval:
    point: float
    lower: float
    upper: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lower:.3f}, {self.upper:.3f}]"

    @property
    def width(self) -> float:
        return self.upper - self.lower


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap confidence interval.

    Seeded, so a reported interval is reproducible.
    """
    stat = statistic or _mean
    if not values:
        return Interval(0.0, 0.0, 0.0)

    rng = random.Random(seed)
    n = len(values)
    samples = sorted(
        stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    lo = samples[int(alpha / 2 * resamples)]
    hi = samples[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(point=stat(values), lower=lo, upper=hi)


@dataclass(slots=True)
class ComparisonResult:
    difference: float
    ci: Interval
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero. Read with the effect size, not alone."""
        return self.ci.lower > 0 or self.ci.upper < 0

    def __str__(self) -> str:
        mark = "*" if self.significant else " "
        return f"Δ={self.difference:+.3f} [{self.ci.lower:+.3f}, {self.ci.upper:+.3f}] p={self.p_value:.4f}{mark}"


def paired_bootstrap(
    system_a: Sequence[float],
    system_b: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> ComparisonResult:
    """Paired bootstrap test on per-item scores.

    Paired on the item, which is why `parallel_id` exists in the dataset schema:
    two systems are compared on *the same questions*, removing question difficulty
    as a source of variance.
    """
    if len(system_a) != len(system_b):
        raise ValueError(f"{len(system_a)} vs {len(system_b)} scores — must be paired")
    if not system_a:
        return ComparisonResult(0.0, Interval(0.0, 0.0, 0.0), 1.0, 0)

    diffs = [a - b for a, b in zip(system_a, system_b, strict=True)]
    observed = _mean(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    resampled = sorted(_mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))

    # Two-sided p: how often a resample lands on the other side of zero.
    crossings = sum(1 for d in resampled if (d <= 0) if observed > 0) or sum(
        1 for d in resampled if d >= 0
    )
    p = min(1.0, 2 * crossings / resamples)

    return ComparisonResult(
        difference=observed,
        ci=Interval(observed, resampled[int(0.025 * resamples)], resampled[int(0.975 * resamples)]),
        p_value=p,
        n=n,
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni correction. Returns name -> survives correction.

    Applied across the six baselines: without it, comparing one system against six
    others makes a spurious 'win' likely by chance alone.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, bool] = {}
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        if p <= threshold and all(out.get(n, True) for n, _ in ordered[:i]):
            out[name] = True
        else:
            out[name] = False
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-language aggregation
# ──────────────────────────────────────────────────────────────────────────────

MetricName = Literal["macro_f1", "binary_f1", "ece", "hallucination_rate", "abstention_rate"]


@dataclass(slots=True)
class LanguageBreakdown:
    """Per-language values with the cross-lingual gap made explicit.

    The gap is computed rather than left for a reader to notice: it is the number
    multilingual comparisons turn on. Report the per-language gap directly
    rather than hiding it in an overall average.
    """

    metric: str
    by_language: dict[Language, float] = field(default_factory=dict)
    reference: Language = "en"
    higher_is_better: bool = True

    def gap(self, language: Language) -> float:
        """Relative shortfall against the reference language."""
        ref = self.by_language.get(self.reference)
        val = self.by_language.get(language)
        if not ref or val is None:
            return 0.0
        return (val - ref) / abs(ref) * (1 if self.higher_is_better else -1)

    @property
    def worst_gap(self) -> tuple[Language, float]:
        gaps = {
            lang: self.gap(lang) for lang in self.by_language if lang != self.reference
        }
        if not gaps:
            return self.reference, 0.0
        return min(gaps.items(), key=lambda kv: kv[1])

    def meets_criterion(self, max_relative_gap: float = 0.10) -> bool:
        if self.reference not in self.by_language:
            return False
        ref = self.by_language[self.reference]
        if ref == 0 and not self.higher_is_better:
            return all(value == 0 for value in self.by_language.values())
        return self.worst_gap[1] >= -max_relative_gap

    def format(self) -> str:
        rows = [f"{self.metric}:"]
        for lang in LANGUAGES:
            if lang not in self.by_language:
                continue
            gap = "" if lang == self.reference else f"  ({self.gap(lang):+.1%} vs {self.reference})"
            rows.append(f"  {lang}  {self.by_language[lang]:.3f}{gap}")
        lang, gap = self.worst_gap
        rows.append(
            f"  worst gap: {lang} {gap:+.1%} — "
            f"{'within' if self.meets_criterion() else 'EXCEEDS'} the 10% criterion"
        )
        return "\n".join(rows)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

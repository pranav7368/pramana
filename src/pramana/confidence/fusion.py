"""Confidence estimation — fuse the available signals into one calibrated score.

Stage 4. The output is a number an operator can route on: send low-confidence
answers to a human, and audit factual reliability over time.

**Why fuse rather than ask the model.** Verbalised confidence diverges from both
token probabilities and actual accuracy, and RAG systems are systematically
overconfident under noisy evidence -- precisely the regime multilingual retrieval
produces. So confidence is computed from independent evidence about the answer,
not from the model's opinion of itself.

**The signals, and what is actually available here.**

| | Signal | Status in this project |
|---|---|---|
| S1 | Retrieval quality | available |
| S2 | Entailment strength | available -- the strongest signal |
| S3 | Token log-probabilities | **unavailable** on every free endpoint measured |
| S4 | Self-consistency | available, but uninformative for short answers |

S3's absence is handled by **refitting without it**, never by imputing a value.
An imputed constant carries no information while appearing to, which corrupts
calibration silently -- the exact failure this module exists to prevent.

**Why logistic regression.** Interpretability is a requirement, not a preference.
In an enterprise deployment someone will ask why a specific answer scored low, and
the answer must be inspectable. The fitted weights are also a research finding in
their own right: which signal dominates, and does that ordering change by language?
A neural fusion model would hide exactly that.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pramana.schemas import (
    Band,
    ConfidenceReport,
    DetectionResult,
    Draft,
    Language,
    RetrievalResult,
)

log = logging.getLogger(__name__)

# Feature order is fixed and explicit. A silent reordering between fitting and
# inference would produce confident nonsense, so the order is stored with the
# model and verified on load.
FEATURE_ORDER: tuple[str, ...] = (
    # S1 — retrieval
    "s1_top_score",
    "s1_score_margin",
    "s1_mean_score",
    "s1_n_chunks",
    # S2 — entailment
    "s2_supported_ratio",
    "s2_contradicted_ratio",
    "s2_unverifiable_ratio",
    "s2_mean_margin",
    "s2_min_margin",
    "s2_n_claims",
    # S3 — generation uncertainty (present only when the provider returns logprobs)
    "s3_mean_logprob",
    "s3_perplexity",
    # S4 — self-consistency
    "s4_self_consistency",
    "s4_dispersion",
    "s4_n_samples",
    "s4_reliable",
    "answer_tokens",
)

S3_FEATURES = frozenset({"s3_mean_logprob", "s3_perplexity"})


def collect_features(
    retrieval: RetrievalResult, draft: Draft, detection: DetectionResult
) -> dict[str, float]:
    """Gather every available signal into one feature dict.

    Features S3 does not provide are **absent**, not zero -- the distinction is
    load-bearing (see module docstring).
    """
    return {**retrieval.signals(), **draft.signals(), **detection.signals()}


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TemperatureCalibrator:
    """Post-hoc calibration by scaling the logit.

    One parameter, fitted by minimising negative log-likelihood on held-out data.
    Chosen because a single parameter cannot overfit the small development sets
    this project will have per language -- isotonic regression is more flexible and
    is compared as an ablation.
    """

    temperature: float = 1.0

    def apply(self, p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        logit = math.log(p / (1 - p))
        return _sigmoid(logit / self.temperature)

    def fit(self, probs: Sequence[float], labels: Sequence[int], *, steps: int = 200) -> None:
        """Grid search over temperature. Deterministic, and fast enough at this scale."""
        if len(probs) != len(labels):
            raise ValueError(f"{len(probs)} probabilities but {len(labels)} labels")
        if not probs:
            raise ValueError("cannot calibrate on an empty set")

        best_t, best_nll = 1.0, float("inf")
        for i in range(1, steps + 1):
            t = i * 5.0 / steps  # search (0, 5]
            nll = self._nll(probs, labels, t)
            if nll < best_nll:
                best_t, best_nll = t, nll
        self.temperature = best_t
        log.info("calibrated temperature=%.3f (NLL=%.4f)", best_t, best_nll)

    @staticmethod
    def _nll(probs: Sequence[float], labels: Sequence[int], t: float) -> float:
        total = 0.0
        for p, y in zip(probs, labels, strict=True):
            p = min(max(p, 1e-6), 1 - 1e-6)
            q = _sigmoid(math.log(p / (1 - p)) / t)
            q = min(max(q, 1e-9), 1 - 1e-9)
            total -= y * math.log(q) + (1 - y) * math.log(1 - q)
        return total / len(probs)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration metrics
# ──────────────────────────────────────────────────────────────────────────────


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> float:
    """ECE — the weighted average gap between confidence and accuracy.

    The project's headline calibration metric, reported **per language**. A score
    well calibrated in English but not in Tamil is a failure of the multilingual
    claim; averaging the two would conceal exactly the thing under study.
    """
    if not probs:
        return 0.0
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} probabilities but {len(labels)} labels")

    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (p, y) for p, y in zip(probs, labels, strict=True) if (lo < p <= hi or (b == 0 and p == 0))
        ]
        if not members:
            continue
        avg_conf = sum(p for p, _ in members) / len(members)
        accuracy = sum(y for _, y in members) / len(members)
        total += (len(members) / len(probs)) * abs(avg_conf - accuracy)
    return total


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def reliability_bins(
    probs: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> list[dict[str, float]]:
    """Data for a reliability diagram. Perfect calibration lies on the diagonal."""
    out: list[dict[str, float]] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (p, y) for p, y in zip(probs, labels, strict=True) if (lo < p <= hi or (b == 0 and p == 0))
        ]
        out.append(
            {
                "bin_lower": lo,
                "bin_upper": hi,
                "count": float(len(members)),
                "mean_confidence": sum(p for p, _ in members) / len(members) if members else 0.0,
                "accuracy": sum(y for _, y in members) / len(members) if members else 0.0,
            }
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fusion model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ConfidenceModel:
    """Logistic regression over the available signals, plus post-hoc calibration.

    Implemented directly rather than via scikit-learn: the model is a dot product
    and a sigmoid, and keeping it dependency-free means confidence estimation runs
    in the base install alongside the offline stub. Feature-name handling — which
    is where a fusion model actually goes wrong — stays explicit and inspectable.
    """

    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    feature_names: tuple[str, ...] = ()
    calibrator: TemperatureCalibrator = field(default_factory=TemperatureCalibrator)
    language: Language | None = None
    bands: tuple[float, float] = (0.75, 0.45)
    """(high, medium) cut-points. Tuned on the dev set, not chosen by intuition."""

    missing_features: tuple[str, ...] = ()
    """Signals unavailable when this model was fitted -- recorded so results can
    state honestly what the score was computed from."""

    stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Per-feature (mean, std) from fitting.

    Must be persisted with the weights and reapplied at inference. Fitting on
    standardised features and scoring on raw ones produces confidently wrong
    numbers with no error -- the weights are in standardised units and the inputs
    are not. Caught by a test, not by anything visible at runtime.
    """

    # ── inference ─────────────────────────────────────────────────────────────

    def score(self, features: dict[str, float]) -> ConfidenceReport:
        if not self.feature_names:
            return self._heuristic(features)

        missing = set(self.feature_names) - set(features)
        if missing:
            report = self._heuristic(features)
            report.missing_signals.append("Fitted model unavailable: missing " + ", ".join(sorted(missing)))
            return report
        x = _standardise(features, self.feature_names, self.stats) if self.stats else features
        z = self.bias + sum(
            self.weights.get(name, 0.0) * x.get(name, 0.0) for name in self.feature_names
        )
        raw = _sigmoid(z)
        calibrated = self.calibrator.apply(raw)

        return ConfidenceReport(
            score=calibrated,
            raw_score=raw,
            band=self.band_for(calibrated),
            signals={k: v for k, v in features.items() if k in self.feature_names},
            calibrator=f"temperature({self.calibrator.temperature:.3f})",
            missing_signals=list(self.missing_features),
        )

    def _heuristic(self, features: dict[str, float]) -> ConfidenceReport:
        """Untrained fallback: a transparent weighted average, dominated by S2.

        Used before any annotated data exists, so the pipeline is runnable end to
        end from day one. Reported with ``calibrator="none"`` and must never be
        presented as a calibrated score.
        """
        supported = features.get("s2_supported_ratio", 0.0)
        contradicted = features.get("s2_contradicted_ratio", 0.0)
        margin = features.get("s2_mean_margin", 0.0)
        retrieval = min(1.0, features.get("s1_top_score", 0.0) * 2)
        consistency = features.get("s4_self_consistency", 0.0)
        s4_usable = features.get("s4_reliable", 0.0) > 0.5

        score = 0.55 * supported + 0.20 * margin + 0.15 * retrieval
        score += 0.10 * consistency if s4_usable else 0.10 * supported
        score -= 0.35 * contradicted  # a contradiction is disqualifying
        score = min(max(score, 0.0), 1.0)

        return ConfidenceReport(
            score=score,
            raw_score=score,
            band=self.band_for(score),
            signals=dict(features),
            calibrator="none",
            missing_signals=["UNTRAINED: heuristic weights, not fitted or calibrated"],
        )

    def band_for(self, score: float) -> Band:
        high, medium = self.bands
        if score >= high:
            return "HIGH"
        return "MEDIUM" if score >= medium else "LOW"

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        feature_rows: Sequence[dict[str, float]],
        labels: Sequence[int],
        *,
        epochs: int = 400,
        lr: float = 0.1,
        l2: float = 0.01,
    ) -> None:
        """Fit by gradient descent on binary cross-entropy.

        The target is human-annotated full faithfulness. Features present in *every*
        row are used; a signal missing anywhere (S3 on a provider without logprobs)
        is dropped from the model entirely rather than filled in.
        """
        if len(feature_rows) != len(labels):
            raise ValueError(f"{len(feature_rows)} rows but {len(labels)} labels")
        if not feature_rows:
            raise ValueError("cannot fit on an empty dataset")

        available = set(feature_rows[0])
        for row in feature_rows[1:]:
            available &= set(row)

        names = tuple(f for f in FEATURE_ORDER if f in available)
        if not names:
            raise ValueError("no feature is present in every row")

        self.feature_names = names
        self.missing_features = tuple(f for f in FEATURE_ORDER if f not in available)
        if S3_FEATURES & set(self.missing_features):
            log.info("fitting without S3 (no logprobs available); using %d features", len(names))

        self.stats = _standardisation_stats(feature_rows, names)
        stats = self.stats
        self.weights = dict.fromkeys(names, 0.0)
        self.bias = 0.0

        for _ in range(epochs):
            grad_w = dict.fromkeys(names, 0.0)
            grad_b = 0.0
            for row, y in zip(feature_rows, labels, strict=True):
                x = _standardise(row, names, stats)
                p = _sigmoid(self.bias + sum(self.weights[n] * x[n] for n in names))
                err = p - y
                for n in names:
                    grad_w[n] += err * x[n]
                grad_b += err

            n_rows = len(feature_rows)
            for n in names:
                self.weights[n] -= lr * (grad_w[n] / n_rows + l2 * self.weights[n])
            self.bias -= lr * grad_b / n_rows

        log.info("fitted confidence model on %d examples, %d features", len(feature_rows), len(names))

    def importance(self) -> list[tuple[str, float]]:
        """Weights by absolute magnitude — which signal matters most.

        A reportable research finding, and the answer to a compliance reviewer
        asking why a particular answer scored low.
        """
        return sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "weights": self.weights,
                    "bias": self.bias,
                    "feature_names": list(self.feature_names),
                    "temperature": self.calibrator.temperature,
                    "language": self.language,
                    "bands": list(self.bands),
                    "missing_features": list(self.missing_features),
                    # Persisted with the weights: without them the model scores
                    # standardised-unit weights against raw inputs.
                    "stats": {k: list(v) for k, v in self.stats.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceModel:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        names = tuple(d["feature_names"])
        unknown = set(names) - set(FEATURE_ORDER)
        if unknown:
            # A model fitted against a different feature set would silently score
            # nonsense; refuse rather than guess.
            raise ValueError(f"model references unknown features: {sorted(unknown)}")
        return cls(
            weights=d["weights"],
            bias=d["bias"],
            feature_names=names,
            calibrator=TemperatureCalibrator(d.get("temperature", 1.0)),
            language=d.get("language"),
            bands=tuple(d.get("bands", (0.75, 0.45))),  # type: ignore[arg-type]
            missing_features=tuple(d.get("missing_features", ())),
            stats={k: (v[0], v[1]) for k, v in (d.get("stats") or {}).items()},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _standardisation_stats(
    rows: Sequence[dict[str, float]], names: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation per feature.

    Necessary because the features are on wildly different scales -- a ratio in
    [0,1] beside a token count in the hundreds. Without it, gradient descent is
    dominated by whichever feature happens to be largest.
    """
    stats: dict[str, tuple[float, float]] = {}
    for n in names:
        values = [r.get(n, 0.0) for r in rows]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        stats[n] = (mean, math.sqrt(var) or 1.0)
    return stats


def _standardise(
    row: dict[str, float], names: Sequence[str], stats: dict[str, tuple[float, float]]
) -> dict[str, float]:
    return {n: (row.get(n, 0.0) - stats[n][0]) / stats[n][1] for n in names}


def confidence_report_summary(report: ConfidenceReport) -> str:
    top = sorted(report.signals.items(), key=lambda kv: -abs(kv[1]))[:3]
    detail = " ".join(f"{k}={v:.2f}" for k, v in top)
    warning = "  [UNCALIBRATED]" if report.calibrator == "none" else ""
    return f"{report.score:.2f} ({report.band}) · {detail}{warning}"

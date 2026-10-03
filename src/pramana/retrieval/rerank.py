"""Optional cross-encoder reranking of the fused candidates.

A cross-encoder reads the query and passage together, so it separates "mentions
the same words" from "answers the question" better than either first-stage
retriever. Multilingual rerankers such as ``BAAI/bge-reranker-v2-m3`` cover
Hindi and Tamil; measure Recall@k per language before and after enabling one.

Scores are squashed to [0, 1]. Downstream thresholds (``weak_retrieval_score``,
signal S1) are written on that scale, and raw cross-encoder logits would
silently move every retrieval across them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CrossEncoderReranker:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512
    batch_size: int = 16
    _model: Any = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Reranking needs sentence-transformers. Install with: pip install -e '.[nlp]'"
            ) from exc
        log.info("loading reranker %s (first use)", self.model_name)
        self._model = CrossEncoder(self.model_name, max_length=self.max_length)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        self._ensure_loaded()
        raw = [float(s) for s in self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)]
        return to_unit_interval(raw)


def to_unit_interval(scores: list[float]) -> list[float]:
    """Probabilities pass through; logits go through a sigmoid.

    Decided per batch, not per score: treating some values in one batch as
    probabilities and others as logits would reorder the passages.
    """
    if all(0.0 <= s <= 1.0 for s in scores):
        return scores
    return [_sigmoid(s) for s in scores]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)

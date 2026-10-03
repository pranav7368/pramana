"""Hybrid retrieval: dense + sparse, fused by Reciprocal Rank Fusion.

RRF combines rankings rather than scores:

    RRF(d) = Σ_r  1 / (k + rank_r(d))

Chosen over weighted score fusion for a reason specific to this project. Dense
cosine similarities and BM25 scores live on incompatible scales, and those scales
shift **per language** -- multilingual embedding similarities are systematically
higher and flatter for Hindi and Tamil than for English. Any fixed score-blending
weight would therefore mean something different in each language, silently
confounding the cross-lingual comparison that RQ4 depends on. Ranks are
scale-free, so one fusion rule behaves identically everywhere.

``k`` (default 60) damps the influence of top ranks: without it, rank 1 would
dominate rank 2 so heavily that the second retriever could never contribute.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from pramana.ingestion.language import detect_language
from pramana.retrieval.sparse import BM25Index
from pramana.schemas import Chunk, Language, RetrievalResult, RetrievedChunk

log = logging.getLogger(__name__)

RRF_K = 60


class DenseIndex(Protocol):
    """Minimal vector-index contract, so FAISS is not a hard dependency."""

    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, top_k: int) -> list[tuple[int, float]]: ...
    @property
    def size(self) -> int: ...


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
    normalise: bool = True,
) -> list[tuple[int, float]]:
    """Fuse ranked lists of ``(index, score)`` into one ranking.

    Only positions are used; the incoming scores are ignored by design.

    ``normalise`` rescales the output to [0, 1] by dividing by the theoretical
    maximum -- the score a document would get by ranking first in every list.

    **Why normalisation is not cosmetic.** Raw RRF scores are tiny: at k=60 a
    rank-1 document scores 1/61 = 0.016. Downstream thresholds are naturally
    written on a 0-1 scale, and comparing a 0.016 against a 0.15 cut-off silently
    classifies *every* retrieval as weak. That bug disabled the entire correction
    stage until an end-to-end test caught it -- the policy never saw retrieval it
    considered good enough to regenerate against.
    """
    w = weights or [1.0] * len(rankings)
    if len(w) != len(rankings):
        raise ValueError(f"got {len(rankings)} rankings but {len(w)} weights")

    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, w, strict=True):
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + weight / (k + rank + 1)

    if normalise and fused:
        maximum = sum(w) / (k + 1)
        if maximum > 0:
            fused = {i: s / maximum for i, s in fused.items()}

    return sorted(fused.items(), key=lambda x: -x[1])


@dataclass(slots=True)
class HybridRetriever:
    """Dense + sparse retrieval over a single-language corpus.

    One retriever per language: mixing languages in one index lets cross-language
    score contamination through and makes per-language retrieval metrics
    impossible to compute cleanly.
    """

    language: Language
    dense: DenseIndex | None = None
    sparse: BM25Index = field(default_factory=BM25Index)
    top_k: int = 5
    candidate_k: int = 20
    """Candidates drawn from each retriever before fusion. Larger than ``top_k``
    so fusion has something to work with -- fusing two lists of 5 mostly
    reproduces whichever list was already right."""

    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    reranker: object | None = None

    def add(self, chunks: list[Chunk]) -> None:
        wrong = [c.chunk_id for c in chunks if c.language != self.language]
        if wrong:
            raise ValueError(
                f"retriever is for {self.language!r} but received chunks in another "
                f"language: {wrong[:3]}. Use one retriever per language."
            )
        self.sparse.add(chunks)
        if self.dense is not None:
            self.dense.add(chunks)

    @property
    def size(self) -> int:
        return self.sparse.size

    @property
    def chunks(self) -> list[Chunk]:
        return self.sparse.chunks

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.top_k
        detection = detect_language(query)

        result = RetrievalResult(
            query=query,
            normalized_query=query.strip(),
            language=self.language,
            script=detection.script,
        )
        if not self.size:
            log.warning("retrieval requested from an empty %s index", self.language)
            return result

        sparse_hits = self.sparse.search(query, top_k=self.candidate_k, language=self.language)
        dense_hits = (
            self.dense.search(query, top_k=self.candidate_k) if self.dense is not None else []
        )

        if not sparse_hits and not dense_hits:
            # A genuinely empty result. Returning it unchanged is the point --
            # the generator must then abstain rather than answer from nothing.
            return result

        rankings = [r for r in (dense_hits, sparse_hits) if r]
        weights = [
            w
            for w, r in ((self.dense_weight, dense_hits), (self.sparse_weight, sparse_hits))
            if r
        ]
        fused = reciprocal_rank_fusion(rankings, weights=weights)[:k]

        dense_scores = dict(dense_hits)
        sparse_scores = dict(sparse_hits)

        result.chunks = [
            RetrievedChunk(
                chunk=self.sparse.chunks[idx],
                rank=rank,
                dense_score=dense_scores.get(idx, 0.0),
                sparse_score=sparse_scores.get(idx, 0.0),
                fused_score=score,
            )
            for rank, (idx, score) in enumerate(fused)
        ]

        if self.reranker is not None:
            self._apply_reranker(query, result)

        return result

    def _apply_reranker(self, query: str, result: RetrievalResult) -> None:
        pairs = [(query, rc.chunk.text) for rc in result.chunks]
        scores = self.reranker.score(pairs)  # type: ignore[attr-defined]
        for rc, s in zip(result.chunks, scores, strict=True):
            rc.rerank_score = float(s)
        result.chunks.sort(key=lambda rc: -rc.score)
        for rank, rc in enumerate(result.chunks):
            rc.rank = rank

    def explain(self, query: str, chunk_id: str) -> dict[str, float]:
        """Per-term BM25 contributions for one chunk -- see ``BM25Index.explain``."""
        for i, c in enumerate(self.sparse.chunks):
            if c.chunk_id == chunk_id:
                return self.sparse.explain(query, i, self.language)
        raise KeyError(f"unknown chunk_id {chunk_id!r}")


@dataclass(slots=True)
class MultilingualRetriever:
    """Routes a query to the retriever for its detected language.

    Language detection happens here rather than inside each retriever so a
    misrouted query fails loudly at the boundary instead of silently searching the
    wrong corpus and returning plausible-looking evidence in the wrong language.
    """

    retrievers: dict[Language, HybridRetriever] = field(default_factory=dict)
    fallback: Language = "en"
    query_variants: Callable[[str, Language], list[str]] | None = None
    """Optional native-script rewrites for Romanised Hindi/Tamil queries
    (see ``query_rewrite``). The original query is always searched as well."""

    def add_language(self, language: Language, retriever: HybridRetriever) -> None:
        if retriever.language != language:
            raise ValueError(
                f"retriever language {retriever.language!r} does not match key {language!r}"
            )
        self.retrievers[language] = retriever

    def retrieve(
        self, query: str, language: Language | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        lang = language or detect_language(query).language
        retriever = self.retrievers.get(lang)

        if retriever is None:
            log.warning(
                "no retriever for language %r; falling back to %r. "
                "Retrieved evidence will be in the wrong language.",
                lang,
                self.fallback,
            )
            retriever = self.retrievers.get(self.fallback)
            if retriever is None:
                raise KeyError(
                    f"no retriever for {lang!r} and no {self.fallback!r} fallback configured"
                )

        from pramana.retrieval.query_rewrite import fuse_results, needs_native_variant

        if self.query_variants is None or not needs_native_variant(query, retriever.language):
            return retriever.retrieve(query, top_k=top_k)
        variants = [v for v in dict.fromkeys(self.query_variants(query, retriever.language)) if v != query]
        if not variants:
            return retriever.retrieve(query, top_k=top_k)
        k = top_k or retriever.top_k
        return fuse_results([retriever.retrieve(v, top_k=k) for v in (query, *variants)], k)

    @property
    def languages(self) -> list[Language]:
        return sorted(self.retrievers)

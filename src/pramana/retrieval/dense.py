"""Optional in-memory semantic index for a small, bounded pilot corpus."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pramana.schemas import Chunk


@dataclass
class SentenceTransformerIndex:
    model_name: str
    min_similarity: float = 0.30
    encoder: Any = None
    vectors: Any = field(default=None, init=False)

    def _encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name, device="cpu")
        return self.encoder

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        encoder = self._encoder()
        texts = [c.text for c in chunks]
        vectors = np.asarray(encoder.encode_documents(texts) if hasattr(encoder, "encode_documents")
                             else encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("Encoder returned invalid document vectors")
        self.vectors = vectors if self.vectors is None else np.concatenate([self.vectors, vectors])

    @property
    def size(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.size or top_k <= 0:
            return []
        encoder = self._encoder()
        vector = np.asarray(encoder.encode_query(query) if hasattr(encoder, "encode_query")
                            else encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0], dtype=np.float32)
        scores = self.vectors @ vector
        if not np.isfinite(scores).all():
            raise ValueError("Encoder returned invalid query vector")
        return [(int(i), float(scores[i])) for i in np.argsort(-scores, kind="stable")[:top_k]
                if scores[i] >= self.min_similarity]

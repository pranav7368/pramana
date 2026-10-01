"""Google text embeddings with bounded requests and no local model weights."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import httpx
import numpy as np

from pramana.generation.base import AuthError, InvalidResponseError, ProviderError, RateLimitError
from pramana.generation.budget import provider_timeout


@dataclass
class GoogleEmbeddingEncoder:
    model: str = "gemini-embedding-001"
    dimensions: int = 768
    timeout: float = 20
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    requests: int = field(default=0, init=False)

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", self.model) or not 1 <= self.dimensions <= 3072:
            raise ValueError("Invalid embedding model or dimensionality")
        if self.model != "gemini-embedding-001":
            raise ValueError("This task-type adapter supports gemini-embedding-001")

    def encode_documents(self, texts):
        return self._encode(list(texts), "RETRIEVAL_DOCUMENT")

    def encode_query(self, query):
        return self._encode([query], "RETRIEVAL_QUERY")[0]

    def _encode(self, texts: list[str], task: str):
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        if any(not isinstance(t, str) or not t.strip() or len(t) > 20000 for t in texts):
            raise ValueError("Embedding texts must be nonempty and at most 20,000 characters")
        key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not key:
            raise AuthError("Google embeddings require GOOGLE_API_KEY", provider="google-embeddings")
        rows = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            body = {"requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                 # The live v1beta endpoint honors these flat fields; nested
                 # embedContentConfig returned 3072 dimensions in our probe.
                 "taskType": task, "outputDimensionality": self.dimensions}
                for t in batch
            ]}
            try:
                with httpx.Client(transport=self.transport, timeout=provider_timeout(self.timeout)) as client:
                    response = client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents",
                        headers={"x-goog-api-key": key}, json=body,
                    )
                    self.requests += 1
            except httpx.HTTPError as exc:
                raise ProviderError("Embedding transport unavailable", provider="google-embeddings") from exc
            if response.status_code in {401, 403}:
                raise AuthError("Embedding credentials rejected", provider="google-embeddings")
            if response.status_code == 429:
                raise RateLimitError("Embedding rate/quota limit", provider="google-embeddings")
            if response.status_code != 200:
                raise ProviderError(f"Embedding service HTTP {response.status_code}", provider="google-embeddings")
            try:
                vectors = np.asarray([r["values"] for r in response.json()["embeddings"]], dtype=np.float32)
            except (ValueError, KeyError, TypeError) as exc:
                raise InvalidResponseError("Malformed embedding response") from exc
            if vectors.shape != (len(batch), self.dimensions) or not np.isfinite(vectors).all():
                raise InvalidResponseError("Embedding shape or values invalid")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise InvalidResponseError("Embedding response contains zero vectors")
            rows.append(vectors / norms)
        return np.concatenate(rows)

"""API vector validation, retrieval integration, and confidential error handling."""
import json

import httpx
import numpy as np
import pytest

from pramana.generation.base import InvalidResponseError, RateLimitError
from pramana.generation.budget import request_budget
from pramana.retrieval.api_embeddings import GoogleEmbeddingEncoder
from pramana.retrieval.dense import SentenceTransformerIndex
from pramana.schemas import Chunk


def test_remote_search_uses_document_query_tasks_without_local_weights(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    tasks = []

    def handle(request):
        rows = json.loads(request.content)["requests"]
        tasks.extend(r["taskType"] for r in rows)
        assert all(r["outputDimensionality"] == 2 for r in rows)
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(200, json={"embeddings": [{"values": [3, 4]} for _ in rows]})

    encoder = GoogleEmbeddingEncoder(dimensions=2, transport=httpx.MockTransport(handle))
    index = SentenceTransformerIndex("remote", encoder=encoder)
    with request_budget(2, 20):
        index.add([Chunk("id", "doc", "The claim deadline is 30 days.", "en")])
        found = index.search("when should I file?", 1)
    assert tasks == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert found[0][0] == 0 and found[0][1] == pytest.approx(1)
    assert np.linalg.norm(index.vectors[0]) == pytest.approx(1)
    assert encoder.requests == 2


@pytest.mark.parametrize("vectors", [[[]], [[0, 0]], [[float("nan"), 1]], [[1, 2], [3, 4]]])
def test_invalid_embeddings_are_rejected(monkeypatch, vectors):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    content = json.dumps({"embeddings": [{"values": r} for r in vectors]})
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    encoder = GoogleEmbeddingEncoder(dimensions=2, transport=transport)
    with pytest.raises(InvalidResponseError):
        encoder.encode_documents(["document"])


def test_embedding_quota_error_does_not_leak_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda _: httpx.Response(429, text="private context and key"))
    with pytest.raises(RateLimitError) as error:
        GoogleEmbeddingEncoder(transport=transport).encode_query("query")
    assert "private" not in str(error.value) and "key" not in str(error.value)

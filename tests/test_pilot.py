"""Pilot boundaries are exercised without live providers or private documents."""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from pramana.api import service
from pramana.config.settings import Settings
from pramana.ingestion.corpus import load_corpus
from pramana.schemas import Action
from tests.test_pipeline import _FixedProvider, build

KEY = "pilot-test-token-0123456789abcdef0123456789"
QUESTION = "When are claims rejected after discharge?"
WRONG = "Claims are rejected if submitted more than 15 days after discharge."
RIGHT = "Claims are rejected if submitted more than 30 days after discharge."


def test_pilot_configuration_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="API_KEY"):
        Settings(mode="pilot")
    with pytest.raises(ValueError, match="CORPUS_DIR"):
        Settings(mode="pilot", api_key=KEY)
    with pytest.raises(ValueError, match="live"):
        Settings(mode="pilot", api_key=KEY, corpus_dir=tmp_path, providers=("stub",))
    cfg = Settings(mode="pilot", api_key=KEY, corpus_dir=tmp_path, providers=("groq",))
    assert KEY not in repr(cfg)


def test_corpus_keeps_document_boundaries_and_versions(tmp_path):
    folder = tmp_path / "en"
    folder.mkdir()
    first = folder / "claims.md"
    first.write_text(RIGHT, encoding="utf-8")
    (folder / "leave.txt").write_text("Employees receive 15 leave days per year.", encoding="utf-8")
    chunks = load_corpus(tmp_path, ("en",))["en"]
    assert len({c.doc_id for c in chunks}) == 2
    assert all(c.metadata["sha256"] and c.metadata["source"] for c in chunks)
    previous = next(c.doc_id for c in chunks if c.metadata["source"].endswith("claims.md"))
    first.write_text(WRONG, encoding="utf-8")
    assert previous not in {c.doc_id for c in load_corpus(tmp_path, ("en",))["en"]}
    with pytest.raises(ValueError, match="size limit"):
        load_corpus(tmp_path, ("en",), max_file_bytes=10)
    with pytest.raises(ValueError, match="language hi"):
        load_corpus(tmp_path, ("hi",))


@pytest.fixture
def pilot_client(monkeypatch, tmp_path):
    cfg = Settings(mode="pilot", api_key=KEY, corpus_dir=tmp_path, providers=("groq",),
                   languages=("en",), expose_corpus=False, cache_enabled=False)
    pipeline = build(_FixedProvider(WRONG, WRONG))
    pipeline.fail_closed = True

    class Router:
        def close(self):
            pass

    monkeypatch.setattr(service.Settings, "from_env", lambda: cfg)
    monkeypatch.setattr(service, "build_pipeline", lambda **kwargs: (pipeline, Router(), False))
    with TestClient(service.app, base_url="http://localhost", raise_server_exceptions=False) as client:
        yield client, pipeline


@pytest.mark.parametrize("path", ["/v1/ready", "/v1/runtime", "/v1/corpus", "/docs", "/openapi.json", "/"])
def test_protected_routes_require_authentication(pilot_client, path):
    client, _ = pilot_client
    assert client.get(path).status_code == 401


def test_both_inference_routes_require_authentication(pilot_client):
    client, _ = pilot_client
    assert client.post("/v1/ask", json={"query": QUESTION}).status_code == 401
    assert client.post("/v1/verify", json={"query": QUESTION, "answer": RIGHT}).status_code == 401
    assert client.get("/v1/health").status_code == 200


def test_authenticated_status_and_corpus_policy(pilot_client):
    client, _ = pilot_client
    headers = {"Authorization": f"Bearer {KEY}"}
    r = client.get("/v1/ready", headers=headers)
    assert r.status_code == 200 and r.json()["mode"] == "pilot"
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-request-id"]
    assert client.get("/v1/corpus", headers=headers).status_code == 404
    assert client.get("/", headers=headers).status_code == 404
    assert client.get("/v1/demo/documents", headers=headers).status_code == 404
    assert client.post("/v1/demo/documents?filename=policy.txt", headers=headers,
                       content=b"policy").status_code == 404


def test_pilot_refuses_to_ship_an_unresolved_contradiction(pilot_client):
    client, _ = pilot_client
    r = client.post("/v1/ask", headers={"Authorization": f"Bearer {KEY}"}, json={"query": QUESTION})
    assert r.status_code == 200
    result = r.json()
    assert result["abstained"] and result["claims"] == []
    assert "15 days" not in result["answer"]
    assert result["rolled_back"] and not result["regressed"]
    assert result["correction_attempts"] == 1


def test_pilot_cannot_disable_assurance_or_search_another_language(pilot_client):
    client, _ = pilot_client
    headers = {"Authorization": f"Bearer {KEY}"}
    assert client.post("/v1/ask", headers=headers, json={"query": QUESTION, "assurance": False}).status_code == 422
    assert client.post("/v1/ask", headers=headers, json={"query": QUESTION, "language": "ta"}).status_code == 422
    assert client.post("/v1/ask", headers=headers, json={"query": "   "}).status_code == 422


def test_body_limit_is_enforced_before_json_parsing(pilot_client):
    client, _ = pilot_client
    response = client.post("/v1/ask", headers={"Authorization": f"Bearer {KEY}"}, content=b"x" * 40000)
    assert response.status_code == 413
    response = client.post("/v1/ask", headers={"Authorization": f"Bearer {KEY}"}, content=iter([b"x" * 20000, b"x" * 20000]))
    assert response.status_code == 413


def test_capacity_rejects_work_and_releases_slot_after_failure(pilot_client):
    client, pipeline = pilot_client
    headers = {"Authorization": f"Bearer {KEY}"}
    slots = service._state["slots"]
    assert slots.acquire(False)
    try:
        assert client.post("/v1/ask", headers=headers, json={"query": QUESTION}).status_code == 503
    finally:
        slots.release()

    class BrokenProvider:
        def generate(self, request):
            raise RuntimeError("sensitive policy data and secret key")

    pipeline.generator.provider = BrokenProvider()
    response = client.post("/v1/ask", headers=headers, json={"query": QUESTION})
    assert response.status_code == 503 and "sensitive" not in response.text
    assert slots.acquire(False), "slot must be released after exceptions"
    slots.release()


def test_api_request_options_do_not_mutate_shared_pipeline(monkeypatch):
    pipeline = build(_FixedProvider(WRONG, WRONG, RIGHT))

    class Router:
        def close(self):
            pass

    monkeypatch.setattr(service.Settings, "from_env", lambda: Settings(offline=True))
    monkeypatch.setattr(service, "build_pipeline", lambda **kwargs: (pipeline, Router(), True))
    with TestClient(service.app, base_url="http://localhost") as client:
        baseline = client.post("/v1/ask", json={"query": QUESTION, "assurance": False, "max_corrections": 0}).json()
        assert baseline["answer"] == WRONG and not baseline["abstained"]
        assert baseline["action_history"] == []
        fixed = client.post("/v1/ask", json={"query": QUESTION}).json()
        assert fixed["answer"] == RIGHT and "REGENERATE" in fixed["action_history"]
    assert pipeline.executor is not None and pipeline.policy.max_iterations == 2


def test_zero_budget_performs_no_revision():
    provider = _FixedProvider(WRONG, RIGHT)
    pipeline = build(provider)
    result = pipeline.run(QUESTION, max_corrections=0)
    assert result.abstained and result.correction_attempts == 0 and provider.calls == 1
    assert result.action_history == [Action.ABSTAIN]


def test_missing_fitted_signal_does_not_silently_impute():
    from pramana.confidence.fusion import ConfidenceModel
    model = ConfidenceModel()
    model.fit([{"s2_supported_ratio": 1.0, "s3_mean_logprob": -0.2},
               {"s2_supported_ratio": 0.0, "s3_mean_logprob": -2.0}], [1, 0])
    report = model.score({"s2_supported_ratio": 1.0})
    assert report.calibrator == "none"
    assert any("s3_mean_logprob" in message for message in report.missing_signals)


def test_retrieval_correction_gets_new_evidence_without_unnecessary_generation():
    from pramana.schemas import Chunk, RetrievalResult, RetrievedChunk
    old = RetrievalResult(QUESTION, QUESTION, "en", "roman", [
        RetrievedChunk(Chunk("old", "d", "The office closes at five.", "en"), 0, fused_score=0.05)
    ])
    fresh = replace(old, chunks=[RetrievedChunk(Chunk("new", "d2", RIGHT, "en"), 0, fused_score=0.9)])

    class Retriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, language, top_k):
            self.calls.append((query, top_k))
            return old if len(self.calls) == 1 else fresh

    provider = _FixedProvider(RIGHT)
    pipeline = build(provider)
    pipeline.retriever = Retriever()
    result = pipeline.run(QUESTION)
    assert Action.RE_RETRIEVE in result.action_history
    assert result.retrieved_chunk_ids == ["new"] and result.evidence_chunk_ids == ["new"]
    assert len(pipeline.retriever.calls) == 2 and provider.calls == 1
    assert pipeline.retriever.calls[1][1] > pipeline.retriever.calls[0][1]


def test_semantic_index_handles_paraphrases_and_rejects_unrelated_vectors():
    import numpy as np

    from pramana.retrieval.dense import SentenceTransformerIndex
    from pramana.schemas import Chunk

    class Encoder:
        def encode(self, texts, **kwargs):
            return np.asarray([[0.0, 1.0] if text == "cafeteria" else [1.0, 0.0] for text in texts])

    index = SentenceTransformerIndex("test", encoder=Encoder())
    index.add([Chunk("a", "d", "submission deadline", "en")])
    assert index.search("when must I file", 5) == [(0, 1.0)]
    assert index.search("cafeteria", 5) == []


def test_conflicting_evidence_uses_matching_probabilities_and_citations():
    from pramana.detection.verifier import GroundingVerifier, NLIScores
    from pramana.schemas import Chunk, Claim, RetrievalResult, RetrievedChunk, Verdict

    class Backend:
        name = "fixed"
        def score(self, premise, hypothesis, language):
            return NLIScores(0.7, 0.1, 0.2) if premise == "support" else NLIScores(0.01, 0.98, 0.01)

    retrieval = RetrievalResult("q", "q", "en", "roman", [
        RetrievedChunk(Chunk("support", "d1", "support", "en"), 0),
        RetrievedChunk(Chunk("conflict", "d2", "conflict", "en"), 1),
    ])
    claims = [Claim("c", "the policy claim", "en")]
    demo = GroundingVerifier(Backend()).verify(claims, retrieval).claim_verdicts[0]
    pilot = GroundingVerifier(Backend(), conflict_policy="contradiction").verify(claims, retrieval).claim_verdicts[0]
    assert demo.verdict is Verdict.SUPPORTED and demo.entailment_prob == pytest.approx(0.7)
    assert demo.supporting_chunk_ids == ["support"]
    assert pilot.verdict is Verdict.CONTRADICTED and pilot.supporting_chunk_ids == ["conflict"]
    assert pilot.contradiction_prob == pytest.approx(0.98)


def test_verification_budget_flags_the_unchecked_answer_tail():
    from pramana.detection.verifier import GroundingVerifier, NLIScores
    from pramana.schemas import Chunk, Claim, RetrievalResult, RetrievedChunk, Verdict

    class Backend:
        name = "counting"
        calls = 0
        def score(self, *args):
            self.calls += 1
            return NLIScores(1, 0, 0)

    backend = Backend()
    verifier = GroundingVerifier(backend, max_claims=2)
    evidence = RetrievalResult("q", "q", "en", "roman", [RetrievedChunk(Chunk("c", "d", "evidence", "en"), 0)])
    result = verifier.verify([Claim(str(i), f"claim {i}", "en") for i in range(5)], evidence)
    assert backend.calls == 2
    assert result.n_claims == 5 and result.count(Verdict.UNVERIFIABLE) == 3


def test_pilot_rejects_nonfinite_runtime_limits():
    with pytest.raises(ValueError, match='positive'):
        Settings(request_budget_s=float('nan'))
    with pytest.raises(ValueError, match='positive'):
        Settings(provider_timeout_s=float('inf'))


def test_pilot_rejects_a_stub_disguised_as_an_approved_provider(monkeypatch, tmp_path):
    import pramana.generation as generation
    from pramana.api.runtime import build_pipeline
    from pramana.generation.registry import ProviderSpec, Registry, RouterPolicy

    registry = Registry(specs={'renamed': ProviderSpec(name='renamed', adapter='stub', default_model='stub')}, policy=RouterPolicy())
    monkeypatch.setattr(generation, 'load_registry', lambda *args: registry)
    cfg = Settings(mode='pilot', api_key=KEY, corpus_dir=tmp_path, providers=('renamed',))
    with pytest.raises(ValueError, match='stub adapters'):
        build_pipeline(settings=cfg)

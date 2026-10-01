"""Tests for the FastAPI service.

Run against the offline stub, so no key and no network are required. The demo is
deliverable D6 and will be shown at the defence — it must not be the least-tested
part of the system.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="API extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

from pramana.api.service import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from pramana.api import service
    from pramana.config.settings import Settings
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service.Settings, "from_env", lambda: Settings(offline=True))
        with TestClient(app, base_url="http://localhost") as c:
            yield c


class TestHealth:
    def test_cross_origin_inference_cannot_spend_demo_api_budget(self, client):
        assert client.post("/v1/ask", json={"query": "claims"},
                           headers={"Origin": "https://attacker.example"}).status_code == 403
        assert client.post("/v1/verify", json={"query": "claims", "answer": "30 days"},
                           headers={"Origin": "https://attacker.example"}).status_code == 403

    def test_reports_ok_and_languages(self, client):
        d = client.get("/v1/health").json()
        assert d["status"] == "ok"
        assert set(d["languages"]) == {"en", "hi", "ta"}

    def test_offline_mode_is_disclosed(self, client):
        """A demo silently serving fixtures would be worse than one that refuses.
        If no key is configured, the response must say so."""
        d = client.get("/v1/health").json()
        if d["offline"]:
            assert "stub" in d["note"].lower() or "fixture" in d["note"].lower()


class TestAsk:
    def test_returns_a_full_assurance_result(self, client):
        r = client.post("/v1/ask", json={"query": "When are claims rejected after discharge?"})
        assert r.status_code == 200
        d = r.json()
        assert {"answer", "confidence", "claims", "action_history", "trace_id"} <= set(d)
        assert d["detected_language"] == "en"

    def test_every_claim_carries_a_verdict_and_evidence_field(self, client):
        """The demo's whole argument is visible here: which claim, which chunk."""
        d = client.post("/v1/ask", json={"query": "When are claims rejected after discharge?"}).json()
        for claim in d["claims"]:
            assert claim["verdict"] in {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}
            assert isinstance(claim["evidence"], list)

    def test_supported_claims_cite_a_real_chunk(self, client):
        corpus = client.get("/v1/corpus").json()
        known = {c["chunk_id"] for chunks in corpus.values() for c in chunks}
        d = client.post("/v1/ask", json={"query": "When are claims rejected after discharge?"}).json()
        for claim in d["claims"]:
            for cid in claim["evidence"]:
                assert cid in known, f"citation {cid} does not exist in the corpus"

    def test_confidence_is_bounded_and_banded(self, client):
        c = client.post("/v1/ask", json={"query": "When are claims rejected after discharge?"}).json()["confidence"]
        assert 0.0 <= c["score"] <= 1.0
        assert c["band"] in {"HIGH", "MEDIUM", "LOW"}

    def test_untrained_confidence_is_disclosed(self, client):
        """An untrained heuristic must never be presented as a calibrated score."""
        c = client.post("/v1/ask", json={"query": "When are claims rejected after discharge?"}).json()["confidence"]
        assert c["calibrator"] in {"none", "n/a"} or c["missing_signals"]

    def test_language_can_be_forced(self, client):
        d = client.post("/v1/ask", json={"query": "claims", "language": "hi"}).json()
        assert d["detected_language"] == "hi"

    def test_hindi_query_is_auto_detected(self, client):
        d = client.post("/v1/ask", json={"query": "क्लेम कब रिजेक्ट होते हैं?"}).json()
        assert d["detected_language"] == "hi"

    def test_unanswerable_query_abstains(self, client):
        d = client.post("/v1/ask", json={"query": "What is the WiFi password in the Chennai office?"}).json()
        assert d["abstained"]
        assert d["claims"] == [], "an abstention must not be decomposed into claims"
        assert d["hallucination_rate"] == 0.0, "declining to answer is not a hallucination"

    def test_empty_query_is_rejected_by_validation(self, client):
        assert client.post("/v1/ask", json={"query": ""}).status_code == 422

    def test_oversized_query_is_rejected(self, client):
        assert client.post("/v1/ask", json={"query": "x" * 5000}).status_code == 422

    def test_correction_budget_is_validated(self, client):
        assert client.post("/v1/ask", json={"query": "q", "max_corrections": 9}).status_code == 422


class TestVerify:
    """The operator surface: audit an answer another system produced."""

    def test_catches_a_planted_numeric_contradiction(self, client):
        r = client.post(
            "/v1/verify",
            json={
                "query": "When are claims rejected?",
                "answer": "Claims are rejected if submitted more than 15 days after discharge.",
            },
        )
        assert r.status_code == 200
        verdicts = [c["verdict"] for c in r.json()["claims"]]
        assert "CONTRADICTED" in verdicts, "the corpus says 30 days, not 15"

    def test_accepts_a_grounded_answer(self, client):
        d = client.post(
            "/v1/verify",
            json={
                "query": "How long do I have to appeal?",
                "answer": "Rejected claims may be appealed within 60 days of the rejection notice.",
            },
        ).json()
        assert "SUPPORTED" in [c["verdict"] for c in d["claims"]]

    def test_flags_an_unsupported_addition(self, client):
        d = client.post(
            "/v1/verify",
            json={
                "query": "What is included?",
                "answer": "The policy includes free gym membership for every employee.",
            },
        ).json()
        assert d["hallucination_rate"] > 0

    def test_does_not_generate(self, client):
        """Verification must return the submitted answer untouched -- it audits,
        it does not rewrite."""
        answer = "Claims are rejected if submitted more than 15 days after discharge."
        d = client.post("/v1/verify", json={"query": "When are claims rejected?", "answer": answer}).json()
        assert d["answer"] == answer


class TestCorpusAndDemo:
    def test_corpus_is_inspectable(self, client):
        """Citations are only checkable if the cited chunks can be read."""
        d = client.get("/v1/corpus").json()
        assert set(d) == {"en", "hi", "ta"}
        assert all(c["chunk_id"] and c["text"] for chunks in d.values() for c in chunks)

    def test_demo_page_is_self_contained(self, client):
        """No CDN, no build step -- a defence presentation is a bad place to
        discover a missing asset."""
        html = client.get("/").text
        assert client.get("/").status_code == 200
        assert "<script" in html and "PRAMANA" in html
        assert "http://" not in html.replace("http://localhost", "")
        assert "cdn." not in html

    def test_demo_page_declares_utf8(self, client):
        """The page renders Devanagari and Tamil."""
        assert 'charset="utf-8"' in client.get("/").text.lower()

    def test_openapi_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        assert "/v1/ask" in schema["paths"] and "/v1/verify" in schema["paths"]

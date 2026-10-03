"""Questions in one language answered from a document in another.

An employee asks in Hindi or Tamil; the policy is in English. Keyword retrieval
shares no tokens across scripts, so the question is routed to the uploaded
document's index and a translated variant is searched alongside the original.
All offline: translations are scripted, no provider is called.
"""
from __future__ import annotations

import pytest

from pramana.generation.base import Completion, GenerationResponse
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever
from pramana.retrieval.query_rewrite import LLMQueryTranslator
from pramana.schemas import Chunk

EN_POLICY = "Employees on approved trips may claim a daily meal allowance of 900 rupees."
HI_QUESTION = "दैनिक भोजन भत्ता कितना है?"
SESSION = {"X-Pramana-Session": "x" * 32}


class _Scripted:
    name = "scripted"

    def __init__(self, *responses: str):
        self.queue = list(responses)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        text = self.queue.pop(0) if self.queue else ""
        return GenerationResponse(completions=[Completion(text=text)], model="m", provider=self.name)

    def capabilities(self, model):
        return set()


def _routed(translate=None) -> MultilingualRetriever:
    english = HybridRetriever(language="en", top_k=3)
    english.add([Chunk("en_0", "d", EN_POLICY, "en"), Chunk("en_1", "d", "Leave policy is separate.", "en")])
    hindi = HybridRetriever(language="hi", top_k=3)
    hindi.add([Chunk("hi_0", "d", "पहले वर्ष में मातृत्व लाभ नहीं मिलता।", "hi")])
    m = MultilingualRetriever(route_to="en", query_translator=translate)
    m.add_language("en", english)
    m.add_language("hi", hindi)
    return m


class TestRouting:
    def test_routed_question_searches_the_document_not_the_question_language(self):
        result = _routed(lambda q, src, dst: ["What is the daily meal allowance?"]).retrieve(
            HI_QUESTION, language="hi")
        assert result.chunk_ids()[0] == "en_0"
        assert result.language == "en"
        assert result.query == HI_QUESTION

    def test_without_translation_a_cross_script_keyword_search_finds_nothing(self):
        assert _routed(None).retrieve(HI_QUESTION, language="hi").is_empty

    def test_translator_is_called_with_question_and_document_languages(self):
        calls = []
        _routed(lambda q, src, dst: calls.append((src, dst)) or []).retrieve(HI_QUESTION, language="hi")
        assert calls == [("hi", "en")]

    def test_same_language_questions_never_call_the_translator(self):
        calls = []
        _routed(lambda q, src, dst: calls.append(q) or []).retrieve("meal allowance", language="en")
        assert calls == []

    def test_unrouted_retriever_keeps_per_language_behaviour(self):
        m = _routed(None)
        m.route_to = None
        assert m.retrieve("मातृत्व लाभ", language="hi").chunk_ids()[0] == "hi_0"


class TestTranslator:
    def test_accepts_output_in_the_target_script_and_memoises(self):
        provider = _Scripted("What is the daily meal allowance?")
        translate = LLMQueryTranslator(provider)
        assert translate(HI_QUESTION, "hi", "en") == ["What is the daily meal allowance?"]
        assert translate(HI_QUESTION, "hi", "en") == ["What is the daily meal allowance?"]
        assert provider.calls == 1

    @pytest.mark.parametrize("reply", ["दैनिक भोजन भत्ता कितना है?", "", "x" * 1000])
    def test_rejects_echoes_empty_and_runaway_output(self, reply):
        assert LLMQueryTranslator(_Scripted(reply))(HI_QUESTION, "hi", "en") == []

    def test_same_language_and_failures_add_no_variant(self):
        provider = _Scripted("anything")
        assert LLMQueryTranslator(provider)("meal allowance", "en", "en") == []
        assert provider.calls == 0

        class Broken(_Scripted):
            def generate(self, request):
                raise RuntimeError("down")

        assert LLMQueryTranslator(Broken())(HI_QUESTION, "hi", "en") == []

    def test_translates_into_native_script_targets(self):
        translate = LLMQueryTranslator(_Scripted("दैनिक भोजन भत्ता कितना है?"))
        assert translate("What is the daily meal allowance?", "en", "hi") == ["दैनिक भोजन भत्ता कितना है?"]


class TestService:
    @pytest.fixture
    def client(self, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from pramana.api import service
        from pramana.config.settings import Settings

        settings = Settings(mode="public", offline=True, trusted_hosts=("localhost",))
        monkeypatch.setattr(service.Settings, "from_env", lambda: settings)
        with TestClient(service.app, base_url="http://localhost") as c:
            yield c

    def upload(self, client, text, name="policy.md"):
        return client.post(f"/v1/demo/documents?filename={name}", content=text.encode(),
                           headers={**SESSION, "Content-Type": "text/markdown"})

    def test_upload_detects_the_document_language(self, client):
        assert self.upload(client, "# यात्रा नीति\n\nदैनिक भोजन भत्ता 900 रुपये है।").json()["language"] == "hi"
        assert self.upload(client, EN_POLICY).json()["language"] == "en"
        info = client.get("/v1/demo/documents", headers=SESSION).json()
        # A new upload replaces the previous one: one active document per visitor.
        assert info["uploaded"]["language"] == "en" and info["uploaded"]["name"] == "policy.md"
        assert info["documents"]["hi"].get("sample") is True

    def test_audit_reports_answer_and_document_languages(self, client):
        self.upload(client, EN_POLICY)
        d = client.post("/v1/verify", json={"query": HI_QUESTION, "answer": "दैनिक भोजन भत्ता 900 रुपये है।"},
                        headers=SESSION).json()
        assert d["detected_language"] == "hi" and d["evidence_language"] == "en"

    def test_without_an_upload_each_language_uses_its_own_sample(self, client):
        d = client.post("/v1/ask", json={"query": "रिजेक्ट क्लेम की अपील कितने दिन में कर सकते हैं?"},
                        headers=SESSION).json()
        assert d["detected_language"] == "hi" and d["evidence_language"] == "hi"

    def test_reset_returns_every_language_to_the_samples(self, client):
        self.upload(client, EN_POLICY)
        assert client.delete("/v1/demo/documents", headers=SESSION).status_code == 200
        assert client.get("/v1/demo/documents", headers=SESSION).json()["uploaded"] is None

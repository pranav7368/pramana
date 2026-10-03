"""Batched verification, the completeness guard, query transliteration,
reranking, calibration intervals and the registry default fix."""

from __future__ import annotations

import json
import logging

import pytest

from pramana.confidence.fusion import (
    adaptive_calibration_error,
    calibration_interval,
    expected_calibration_error,
)
from pramana.correction.policy import CorrectionExecutor, CorrectionPolicy, VerdictProfile
from pramana.detection.verifier import (
    GroundingVerifier,
    KeywordNLIBackend,
    LLMNLIBackend,
    NLIScores,
    parse_batch_labels,
)
from pramana.generation.base import Completion, GenerationResponse
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever
from pramana.retrieval.query_rewrite import (
    LLMQueryTransliterator,
    fuse_results,
    needs_native_variant,
)
from pramana.retrieval.rerank import to_unit_interval
from pramana.schemas import (
    Action,
    Chunk,
    Claim,
    ClaimVerdict,
    DetectionResult,
    Draft,
    RetrievalResult,
    RetrievedChunk,
    Verdict,
)


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


def _retrieval(*texts: str, language="en") -> RetrievalResult:
    return RetrievalResult(
        query="q", normalized_query="q", language=language, script="native",
        chunks=[RetrievedChunk(chunk=Chunk(f"c{i}", "d", t, language), rank=i, fused_score=1 - i / 10)
                for i, t in enumerate(texts)],
    )


# ── batched verification ──────────────────────────────────────────────────────


class TestBatchParsing:
    def test_accepts_one_label_per_premise_in_any_order(self):
        assert parse_batch_labels("2: NEUTRAL\n1: entailment\n3) CONTRADICTION", 3) == [
            "ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

    @pytest.mark.parametrize("reply", [
        "1: ENTAILMENT",                                # missing premise
        "1: ENTAILMENT\n1: NEUTRAL",                    # duplicate
        "1: ENTAILMENT\n3: NEUTRAL",                    # out of range
        "1: ENTAILMENT\nPremise 2 looks NEUTRAL",       # prose
    ])
    def test_rejects_any_ambiguous_alignment(self, reply):
        assert parse_batch_labels(reply, 2) is None


class TestBatchedVerifier:
    def test_one_request_per_claim_with_per_chunk_citations(self):
        provider = _Scripted("1: NEUTRAL\n2: ENTAILMENT\n3: NEUTRAL")
        verifier = GroundingVerifier(backend=LLMNLIBackend(provider))
        result = verifier.verify([Claim("a", "Claims close after 30 days.", "en")],
                                 _retrieval("Leave policy.", "Claims close after 30 days.", "Parking."))
        assert provider.calls == 1
        verdict = result.claim_verdicts[0]
        assert verdict.verdict is Verdict.SUPPORTED
        assert verdict.supporting_chunk_ids == ["c1"]

    def test_malformed_batch_falls_back_to_per_chunk_requests(self):
        provider = _Scripted("I think premise two.", "NEUTRAL", "CONTRADICTION")
        verifier = GroundingVerifier(backend=LLMNLIBackend(provider))
        result = verifier.verify([Claim("a", "Claims close after 15 days.", "en")],
                                 _retrieval("Leave policy.", "Claims close after 30 days."))
        assert provider.calls == 3
        assert result.claim_verdicts[0].verdict is Verdict.CONTRADICTED
        assert result.claim_verdicts[0].supporting_chunk_ids == ["c1"]

    def test_batch_can_be_disabled_for_the_ablation(self):
        provider = _Scripted("NEUTRAL", "ENTAILMENT")
        GroundingVerifier(backend=LLMNLIBackend(provider, batch=False)).verify(
            [Claim("a", "x", "en")], _retrieval("p1", "p2"))
        assert provider.calls == 2

    def test_backend_returning_wrong_count_fails_loudly(self):
        class Short:
            name = "short"

            def score(self, premise, hypothesis, language):
                return NLIScores(0, 0, 1)

            def score_batch(self, premises, hypothesis, language):
                return [NLIScores(1, 0, 0)]

        with pytest.raises(ValueError, match="1 scores for 2 chunks"):
            GroundingVerifier(backend=Short()).verify([Claim("a", "x", "en")], _retrieval("p1", "p2"))

    def test_backends_without_batch_support_still_work(self):
        result = GroundingVerifier(backend=KeywordNLIBackend()).verify(
            [Claim("a", "Claims close after 30 days.", "en")],
            _retrieval("Claims close after 30 days.", "Unrelated."))
        assert result.claim_verdicts[0].verdict is Verdict.SUPPORTED


# ── completeness guard ────────────────────────────────────────────────────────


def _detection(supported: int, unverifiable: int = 0) -> DetectionResult:
    verdicts = [Verdict.SUPPORTED] * supported + [Verdict.UNVERIFIABLE] * unverifiable
    return DetectionResult([
        ClaimVerdict(claim=Claim(str(i), f"claim {i}", "en"), verdict=v,
                     entailment_prob=0.9 if v is Verdict.SUPPORTED else 0.1,
                     contradiction_prob=0.05, neutral_prob=0.05 if v is Verdict.SUPPORTED else 0.85)
        for i, v in enumerate(verdicts)
    ])


class TestCompletenessGuard:
    def test_a_revision_that_says_less_is_not_an_improvement(self):
        before = _detection(3, 1)
        executor = CorrectionExecutor(provider=_Scripted("Shorter answer."))
        outcome = executor.apply(Action.REGENERATE, Draft("Long answer.", "en"), before, _retrieval("e"),
                                 reverify=lambda text: _detection(1, 0), language="en")
        assert VerdictProfile.of(_detection(1)).is_better_than(VerdictProfile.of(before))
        assert not outcome.accepted

    def test_a_revision_that_keeps_grounded_content_is_accepted(self):
        executor = CorrectionExecutor(provider=_Scripted("Better answer."))
        outcome = executor.apply(Action.REGENERATE, Draft("Answer.", "en"), _detection(3, 1), _retrieval("e"),
                                 reverify=lambda text: _detection(3, 0), language="en")
        assert outcome.accepted

    def test_guard_can_be_disabled_for_the_ablation(self):
        executor = CorrectionExecutor(provider=_Scripted("Shorter."),
                                      policy=CorrectionPolicy(preserve_supported=False))
        outcome = executor.apply(Action.REGENERATE, Draft("Long.", "en"), _detection(3, 1), _retrieval("e"),
                                 reverify=lambda text: _detection(1, 0), language="en")
        assert outcome.accepted


# ── Romanised queries ─────────────────────────────────────────────────────────


HI_CORPUS = "क्लेम डिस्चार्ज के 30 दिनों के अंदर जमा करना होगा। देर से जमा किया गया क्लेम रिजेक्ट होता है।"


def _hindi_retriever(rewrite) -> MultilingualRetriever:
    r = HybridRetriever(language="hi", top_k=3)
    r.add([Chunk("hi_0", "d", HI_CORPUS, "hi"), Chunk("hi_1", "d", "छुट्टी की नीति अलग दस्तावेज़ में है।", "hi")])
    m = MultilingualRetriever(query_variants=rewrite)
    m.add_language("hi", r)
    return m


class TestRomanisedQueries:
    def test_detects_only_roman_hindi_and_tamil(self):
        assert needs_native_variant("claim kab tak submit karna hai", "hi")
        assert not needs_native_variant("क्लेम कब तक जमा करना है", "hi")
        assert not needs_native_variant("when is the claim due", "en")

    def test_native_variant_recovers_evidence_the_roman_query_misses(self):
        query = "claim kab tak submit karna hai"
        assert _hindi_retriever(None).retrieve(query, language="hi").is_empty
        result = _hindi_retriever(lambda q, lang: ["क्लेम कब तक जमा करना है"]).retrieve(query, language="hi")
        assert result.chunk_ids()[0] == "hi_0"
        assert result.query == query
        assert "क्लेम" in result.normalized_query
        assert 0 < result.top_score <= 1

    def test_native_queries_never_call_the_rewriter(self):
        calls = []
        _hindi_retriever(lambda q, lang: calls.append(q) or []).retrieve("क्लेम कब तक", language="hi")
        assert calls == []

    def test_transliterator_rejects_output_in_the_wrong_script_and_memoises(self):
        provider = _Scripted("claim kab tak")
        rewrite = LLMQueryTransliterator(provider)
        assert rewrite("claim kab tak", "hi") == []
        assert rewrite("claim kab tak", "hi") == []
        assert provider.calls == 1

    def test_transliterator_failure_degrades_to_no_variant(self):
        class Broken(_Scripted):
            def generate(self, request):
                raise RuntimeError("down")

        assert LLMQueryTransliterator(Broken())("claim kab tak", "hi") == []

    def test_fusion_normalises_and_deduplicates(self):
        a, b = _retrieval("x", "y"), _retrieval("y", "z")
        fused = fuse_results([a, b], top_k=5)
        assert sorted(fused.chunk_ids()) == ["c0", "c1"]
        assert fused.top_score == pytest.approx(1.0) or fused.top_score <= 1.0


# ── reranking ─────────────────────────────────────────────────────────────────


class TestRerankScale:
    def test_probabilities_pass_through(self):
        assert to_unit_interval([0.2, 0.9]) == [0.2, 0.9]

    def test_logits_are_squashed_consistently_for_the_whole_batch(self):
        scores = to_unit_interval([-3.0, 0.5, 4.0])
        assert all(0 < s < 1 for s in scores)
        assert scores == sorted(scores)

    def test_reranker_order_drives_retrieval(self):
        class Reverse:
            def score(self, pairs):
                return [i / 10 for i in range(len(pairs))]

        r = HybridRetriever(language="en", top_k=2, reranker=Reverse())
        r.add([Chunk("a", "d", "claims close after thirty days", "en"),
               Chunk("b", "d", "claims office opening hours", "en")])
        result = r.retrieve("claims close")
        assert result.top_score <= 1 and result.chunks[0].rerank_score == max(c.rerank_score for c in result.chunks)


# ── calibration reporting ─────────────────────────────────────────────────────


class TestCalibrationReporting:
    def test_adaptive_ece_is_zero_for_perfect_calibration(self):
        probs = [0.0] * 5 + [1.0] * 5
        labels = [0] * 5 + [1] * 5
        assert adaptive_calibration_error(probs, labels) == 0.0

    def test_adaptive_ece_sees_clustered_miscalibration(self):
        probs = [0.9] * 10
        labels = [1] * 5 + [0] * 5
        assert adaptive_calibration_error(probs, labels) == pytest.approx(0.4)
        assert expected_calibration_error(probs, labels) == pytest.approx(0.4)

    def test_interval_brackets_the_point_and_is_reproducible(self):
        probs = [0.1, 0.4, 0.6, 0.8, 0.9, 0.3, 0.7, 0.2]
        labels = [0, 0, 1, 1, 1, 1, 0, 0]
        first = calibration_interval(probs, labels, resamples=300)
        assert first["lower"] <= first["point"] <= first["upper"]
        assert first == calibration_interval(probs, labels, resamples=300)


# ── registry ──────────────────────────────────────────────────────────────────


def test_registry_without_failover_list_uses_the_default(tmp_path):
    from pramana.generation.registry import Registry, RouterPolicy

    path = tmp_path / "providers.yaml"
    path.write_text("providers:\n  stub:\n    adapter: stub\nrouter:\n  strategy: priority_with_failover\n",
                    encoding="utf-8")
    assert Registry.load(path).policy.failover_on == RouterPolicy().failover_on


# ── observability ─────────────────────────────────────────────────────────────


def test_json_log_lines_carry_the_request_id_and_never_a_traceback():
    from pramana.api.observability import JsonFormatter, request_id_var

    token = request_id_var.set("abc123")
    try:
        try:
            raise ValueError("secret document text")
        except ValueError:
            record = logging.LogRecord("pramana", logging.ERROR, __file__, 1, "failed", None,
                                       exc_info=__import__("sys").exc_info())
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["request_id"] == "abc123"
    assert line["exception"] == "ValueError"
    assert "secret" not in json.dumps(line)


# ── HTTP surface ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pramana.api import service
    from pramana.config.settings import Settings

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service.Settings, "from_env", lambda: Settings(offline=True))
        with TestClient(service.app, base_url="http://localhost") as c:
            yield c


class TestHttpSurface:
    def test_demo_page_pins_its_script_by_hash(self, client):
        import base64
        import hashlib
        import re

        response = client.get("/")
        csp = response.headers["content-security-policy"]
        body = re.search(r"<script>(.*?)</script>", response.text, flags=re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp
        assert "unsafe-eval" not in csp and "default-src 'none'" in csp
        assert response.headers.get_list("content-security-policy") == [csp]

    def test_json_responses_get_a_deny_all_policy(self, client):
        assert client.get("/v1/health").headers["content-security-policy"].startswith("default-src 'none'")

    def test_metrics_count_requests_by_known_route_only(self, client):
        client.get("/v1/health")
        client.get("/definitely/not/a/route")
        body = client.get("/v1/metrics").text
        assert 'route="/v1/health",status="200"' in body
        assert 'route="other",status="404"' in body
        assert "/definitely" not in body

    def test_ready_reports_retrieval_upgrades(self, client):
        ready = client.get("/v1/ready").json()
        assert ready["reranker_model"] is None
        assert ready["query_transliteration"] is False  # offline: no provider to rewrite with

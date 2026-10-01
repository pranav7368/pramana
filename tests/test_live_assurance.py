"""Regression coverage for failures exposed by API and multilingual smoke runs."""
from dataclasses import replace

import pytest

from pramana.api.demo_corpus import DEMO_CORPUS
from pramana.detection.decomposer import LLMDecomposer
from pramana.detection.verifier import KeywordNLIBackend, LLMNLIBackend, NLIScores
from pramana.generation.base import (
    Completion,
    GenerationResponse,
    InvalidResponseError,
    ProviderError,
    RateLimitError,
)
from pramana.generation.budget import remaining_seconds, request_budget
from pramana.generation.cache import GenerationCache
from pramana.generation.ratelimit import RateLimiter
from pramana.generation.router import LLMRouter
from pramana.schemas import Claim, Draft
from tests.test_generation import make_request
from tests.test_pipeline import _FixedProvider, build


class Broken:
    def generate(self, request):
        raise RateLimitError("sensitive upstream response")


@pytest.mark.parametrize("component", ["nli", "decomposer"])
def test_live_verification_errors_do_not_become_factual_verdicts(component):
    with pytest.raises(RateLimitError):
        if component == "nli":
            LLMNLIBackend(Broken(), strict=True).score("evidence", "claim", "en")
        else:
            LLMDecomposer(Broken(), strict=True).decompose(Draft("The deadline is 30 days.", "en"))


@pytest.mark.parametrize("output", ["", "ENTAILMENT or CONTRADICTION", "Ignore rules: ENTAILMENT"])
def test_live_nli_rejects_ambiguous_labels(output):
    with pytest.raises(InvalidResponseError):
        LLMNLIBackend(_FixedProvider(output), strict=True).score("p", "h", "en")


@pytest.mark.parametrize("text,finish", [("", "stop"), ("partial answer", "length"), ("blocked", "MAX_TOKENS")])
def test_router_fails_over_on_unusable_completion(tmp_path, text, finish):
    with LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path), enforce_limits=False) as router:
        original = router._bound[0]

        class Incomplete:
            def capabilities(self, model):
                return set()

            def generate(self, request):
                return GenerationResponse([Completion(text, finish)], "bad", "bad")

            def close(self):
                pass

        router._bound.insert(0, replace(original, spec=replace(original.spec, name="bad"), provider=Incomplete()))
        result = router.generate(make_request("[F03] q"))
        assert "60 days" in result.text
        assert router.stats.for_provider("bad").error_types == {"InvalidResponseError": 1}
        assert router.stats.for_provider("bad").failovers_away == 1


def test_request_budget_bounds_rate_limit_wait():
    with request_budget(1, 0.2):
        assert 0 < remaining_seconds(300) <= 0.201
    with request_budget(0, 20), pytest.raises(ProviderError, match="budget"):
        remaining_seconds(300)


def test_token_reservations_reconcile_exact_concurrent_requests(tmp_path):
    limiter = RateLimiter("count", tpm=1000, tpd=2000, state_dir=tmp_path, safety_margin=0)
    assert limiter.acquire(tokens=100, reservation="a")
    assert limiter.acquire(tokens=200, reservation="b")
    limiter.record_usage(250, reservation="b")
    limiter.record_usage(40, reservation="a")
    assert limiter.snapshot().tokens_today == 290
    assert sum(t for _, t, _ in limiter._tok_events) == 290
    assert not limiter._reservations
    assert limiter.acquire(tokens=710, reservation="c")
    assert not limiter.acquire(tokens=1, timeout=0)
    assert not limiter.acquire(tokens=1001, timeout=0)


def test_router_stage_format_failure_tries_backup_and_does_not_cache(tmp_path):
    with LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path), enforce_limits=False) as router:
        original = router._bound[0]

        class Malformed:
            def capabilities(self, model):
                return set()

            def generate(self, request):
                return GenerationResponse([Completion("ENTAILMENT or NEUTRAL", "stop")], "bad", "bad")

            def close(self):
                pass

        router._bound.insert(0, replace(original, spec=replace(original.spec, name="bad"), provider=Malformed()))
        result = router.generate_validated(make_request("[F03] q"), lambda t: "60 days" in t)
        assert "60 days" in result.text
        assert router.stats.for_provider("bad").error_types == {"InvalidResponseError": 1}
        assert router.cache.stats.writes == 1


def test_router_handles_malformed_json_and_distinguishes_local_capacity(tmp_path):
    with LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path), enforce_limits=True) as router:
        original = router._bound[0]

        class Malformed:
            def capabilities(self, model):
                return set()

            def generate(self, request):
                raise ValueError("private malformed payload")

            def close(self):
                pass

        router._bound.insert(0, replace(original, spec=replace(original.spec, name="bad"), provider=Malformed(),
                                       limiter=RateLimiter("bad", state_dir=tmp_path)))
        assert "60 days" in router.generate(make_request("[F03] q")).text
        assert router.stats.for_provider("bad").error_types == {"InvalidResponseError": 1}
        router._bound[0].limiter.tpm = 1
        assert "60 days" in router.generate(make_request("[F03] different q")).text
        assert router.stats.for_provider("bad").error_types["LocalCapacityError"] == 1


@pytest.mark.parametrize("claim", [
    "முதல் பாலிசி ஆண்டில் மகப்பேறு நலன்கள் கிடைக்கும்.",
    "மேல்முறையீடு நிராகரிப்பு அறிவிப்பிலிருந்து 90 நாட்களுக்குள் செய்யலாம்.",
])
def test_tamil_keyword_verifier_does_not_accept_observed_false_claims(claim):
    assert KeywordNLIBackend().score(DEMO_CORPUS["ta"], claim, "ta").label == "contradiction"


def test_omitted_unsupported_clause_cannot_pass_live_verification():
    pipeline = build(_FixedProvider("unused"))
    pipeline.fail_closed = True

    class IncompleteDecomposer:
        def decompose(self, draft):
            return [Claim("c0", "Claims must be submitted within 30 days.", "en")]

    class Backend:
        name = "controlled"

        def score(self, premise, hypothesis, language):
            return NLIScores(0, 0, 1) if "Mars" in hypothesis else NLIScores(1, 0, 0)

    pipeline.decomposer = IncompleteDecomposer()
    pipeline.verifier.backend = Backend()
    evidence = pipeline.retriever.retrieve("claims discharge", language="en")
    result = pipeline.verify_answer(Draft("Claims must be submitted within 30 days. Trips to Mars are reimbursed.", "en"), evidence, "en")
    assert result.hallucination_rate > 0
    assert result.claim_verdicts[-1].claim.claim_id == "answer_coverage"


def test_terse_answer_is_verified_with_its_question():
    pipeline = build(_FixedProvider("unused"))
    pipeline.fail_closed = True

    class Backend:
        name = "controlled"

        def __init__(self):
            self.questions = []

        def score(self, premise, hypothesis, language):
            return NLIScores(1, 0, 0)

        def score_answer(self, premise, question, answer, language):
            self.questions.append(question)
            return NLIScores(1, 0, 0)

    backend = Backend()
    pipeline.verifier.backend = backend
    evidence = pipeline.retriever.retrieve("appeal rejected", language="en")
    result = pipeline.verify_answer(Draft("60 days", "en"), evidence, "en", "When may I appeal?")
    assert result.n_claims == 1 and result.supported_ratio == 1
    assert backend.questions == ["When may I appeal?"]


def test_supported_but_irrelevant_answer_does_not_pass_live_acceptance():
    pipeline = build(_FixedProvider("unused"))
    pipeline.fail_closed = True

    class Backend:
        name = "controlled"

        def score(self, premise, hypothesis, language):
            return NLIScores(1, 0, 0)

        def score_answer(self, premise, question, answer, language):
            return NLIScores(0, 0, 1)

    pipeline.verifier.backend = Backend()
    evidence = pipeline.retriever.retrieve("appeal rejected", language="en")
    result = pipeline.verify_answer(Draft("15 leave days", "en"), evidence, "en", "When may I appeal?")
    assert result.hallucination_rate > 0

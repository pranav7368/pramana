"""Tests for the provider-agnostic generation layer.

Everything here runs offline: no API key, no network, no model download. That is
the point -- the layer must be verifiable on the dev laptop.
"""

from __future__ import annotations

import pytest

from pramana.generation import (
    Capability,
    GenerationCache,
    GenerationRequest,
    LLMRouter,
    Message,
    ProviderError,
    Registry,
    StubProvider,
    load_registry,
    request_fingerprint,
)
from pramana.generation.base import (
    Completion,
    GenerationResponse,
    QuotaExhaustedError,
    RateLimitError,
    TokenLogProb,
    Usage,
)
from pramana.generation.ratelimit import RateLimiter
from pramana.generation.stub import FIXTURES, FIXTURES_BY_ID


def make_request(text: str = "hello", **kw) -> GenerationRequest:
    defaults = dict(
        messages=(Message("system", "be terse"), Message("user", text)),
        model="stub-v1",
    )
    defaults.update(kw)
    return GenerationRequest(**defaults)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — the ground truth the whole pipeline is tested against
# ──────────────────────────────────────────────────────────────────────────────


class TestFixtures:
    def test_all_languages_represented(self):
        assert {f.language for f in FIXTURES} == {"en", "hi", "ta"}

    def test_claim_and_verdict_counts_align(self):
        # Enforced in Fixture.__post_init__; assert it holds for the shipped set.
        for f in FIXTURES:
            assert len(f.expected_claims) == len(f.expected_verdicts), f.fixture_id

    def test_verdict_vocabulary_is_closed(self):
        allowed = {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}
        for f in FIXTURES:
            assert set(f.expected_verdicts) <= allowed, f.fixture_id

    def test_action_vocabulary_matches_policy_table(self):
        allowed = {"ACCEPT", "REGENERATE", "RE_RETRIEVE", "PRUNE", "ABSTAIN"}
        for f in FIXTURES:
            assert f.expected_action in allowed, f.fixture_id

    def test_every_defect_class_is_covered(self):
        """Detection must be exercised on all three hallucination types plus a
        negative control -- otherwise a detector that flags everything passes."""
        verdicts = {v for f in FIXTURES for v in f.expected_verdicts}
        assert verdicts == {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}
        assert any(f.expected_action == "ACCEPT" for f in FIXTURES), "no negative control"
        assert any(f.expected_action == "ABSTAIN" for f in FIXTURES), "no empty-retrieval case"

    def test_empty_retrieval_fixture_has_no_context(self):
        f = FIXTURES_BY_ID["F04"]
        assert f.context == []
        assert all(v == "UNVERIFIABLE" for v in f.expected_verdicts)

    def test_indic_fixtures_carry_native_script(self):
        assert any("ऀ" <= c <= "ॿ" for c in FIXTURES_BY_ID["F06"].answer), "no Devanagari"
        assert any("஀" <= c <= "௿" for c in FIXTURES_BY_ID["F07"].answer), "no Tamil"


# ──────────────────────────────────────────────────────────────────────────────
# Stub provider
# ──────────────────────────────────────────────────────────────────────────────


class TestStubProvider:
    def test_matches_by_fixture_id(self):
        p = StubProvider()
        r = p.generate(make_request("[F05] Is maternity covered in the first year?"))
        assert r.text == FIXTURES_BY_ID["F05"].answer

    def test_matches_by_query_text(self):
        p = StubProvider()
        f = FIXTURES_BY_ID["F03"]
        assert p.generate(make_request(f.query)).text == f.answer

    def test_is_deterministic(self):
        p = StubProvider()
        req = make_request("[F01] why rejected")
        assert p.generate(req).text == p.generate(req).text

    def test_strict_mode_rejects_unmatched_prompt(self):
        with pytest.raises(ProviderError, match="no fixture matches"):
            StubProvider(strict=True).generate(make_request("something entirely unrelated 12345"))

    def test_lenient_mode_echoes_instead_of_raising(self):
        r = StubProvider().generate(make_request("something entirely unrelated 12345"))
        assert "[stub]" in r.text

    def test_multi_sample_returns_n_completions(self):
        r = StubProvider().generate(make_request("[F01] q", n=4))
        assert len(r.completions) == 4

    def test_samples_vary_so_self_consistency_is_measurable(self):
        """Signal S4 needs non-degenerate samples; identical text would make the
        self-consistency feature constant and useless."""
        r = StubProvider().generate(make_request("[F01] q", n=3))
        assert len({c.text for c in r.completions}) > 1

    def test_logprobs_only_when_requested(self):
        assert StubProvider().generate(make_request("[F01] q")).has_logprobs is False
        assert StubProvider().generate(make_request("[F01] q", want_logprobs=True)).has_logprobs

    def test_mean_logprob_is_negative_and_finite(self):
        r = StubProvider().generate(make_request("[F01] q", want_logprobs=True))
        mlp = r.completions[0].mean_logprob
        assert mlp is not None and -5.0 < mlp < 0.0

    def test_mean_logprob_is_none_without_logprobs(self):
        """Signal S3 must be *absent*, not zero -- the fusion model refits without
        it rather than imputing, which would corrupt calibration."""
        assert Completion(text="x").mean_logprob is None


# ──────────────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────────────


class TestCache:
    def test_identical_requests_share_a_key(self):
        a = request_fingerprint(make_request(), "groq", "m")
        b = request_fingerprint(make_request(), "groq", "m")
        assert a == b

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": 0.7},
            {"n": 3},
            {"seed": 42},
            {"max_tokens": 128},
            {"top_p": 0.9},
            {"want_logprobs": True},
            {"stop": ("###",)},
        ],
    )
    def test_output_affecting_params_change_the_key(self, kwargs):
        base = request_fingerprint(make_request(), "groq", "m")
        assert request_fingerprint(make_request(**kwargs), "groq", "m") != base

    def test_prompt_change_changes_the_key(self):
        a = request_fingerprint(make_request("one"), "groq", "m")
        b = request_fingerprint(make_request("two"), "groq", "m")
        assert a != b

    def test_provider_is_part_of_the_key(self):
        """The same nominal model on two hosts can differ -- quantisation, serving
        stack, silent version drift -- so they must not share a cache entry."""
        a = request_fingerprint(make_request(), "groq", "llama-3.3-70b")
        b = request_fingerprint(make_request(), "openrouter", "llama-3.3-70b")
        assert a != b

    def test_round_trip_preserves_content(self, tmp_path):
        cache = GenerationCache(tmp_path)
        original = GenerationResponse(
            completions=[
                Completion(
                    text="नमस्ते दुनिया",
                    finish_reason="stop",
                    token_logprobs=[TokenLogProb("नम", -0.5), TokenLogProb("स्ते", -0.25)],
                )
            ],
            model="m",
            provider="p",
            usage=Usage(prompt_tokens=7, completion_tokens=3),
        )
        cache.put("k" * 64, original)
        loaded = cache.get("k" * 64)

        assert loaded is not None
        assert loaded.text == "नमस्ते दुनिया"  # Unicode survives the JSON round trip
        assert loaded.cached is True
        assert loaded.usage.total_tokens == 10
        assert loaded.completions[0].mean_logprob == pytest.approx(-0.375)

    def test_miss_returns_none_and_counts(self, tmp_path):
        cache = GenerationCache(tmp_path)
        assert cache.get("f" * 64) is None
        assert cache.stats.misses == 1

    def test_corrupt_entry_degrades_to_miss(self, tmp_path):
        """A truncated write must never crash a twelve-hour batch job."""
        cache = GenerationCache(tmp_path)
        key = "c" * 64
        path = cache._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")

        assert cache.get(key) is None
        assert cache.stats.errors == 1
        assert not path.exists(), "corrupt entry should be removed"

    def test_disabled_cache_is_inert(self, tmp_path):
        cache = GenerationCache(tmp_path, enabled=False)
        cache.put("a" * 64, GenerationResponse(completions=[Completion("x")], model="m", provider="p"))
        assert cache.get("a" * 64) is None


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_ships_with_a_loadable_registry(self):
        r = load_registry()
        assert r.specs and "stub" in r.specs and "groq" in r.specs

    def test_stub_is_always_usable_without_credentials(self):
        assert any(s.name == "stub" for s in load_registry().usable())

    def test_model_aliases_resolve(self):
        spec = load_registry().get("groq")
        assert spec.resolve_model("llama3-70b") == "llama-3.3-70b-versatile"

    def test_unknown_model_passes_through(self):
        """An id not in the alias table must reach the provider untouched, so new
        models are usable without a registry edit."""
        spec = load_registry().get("groq")
        assert spec.resolve_model("some-new-model-v9") == "some-new-model-v9"

    def test_empty_model_falls_back_to_default(self):
        spec = load_registry().get("groq")
        assert spec.resolve_model("") == spec.default_model

    def test_usable_is_priority_ordered(self):
        specs = load_registry().usable()
        assert [s.priority for s in specs] == sorted(s.priority for s in specs)

    def test_unknown_provider_raises_with_a_useful_message(self):
        with pytest.raises(KeyError, match="unknown provider"):
            load_registry().get("does-not-exist")

    def test_diagnose_covers_every_entry(self):
        r = load_registry()
        assert len(r.diagnose()) == len(r.specs)

    def test_build_returns_a_working_provider(self):
        provider = load_registry().build("stub")
        assert provider.name == "stub"
        assert provider.health_check() is True

    def test_probe_results_override_declared_capabilities(self, tmp_path):
        """Measurement beats documentation: a provider that advertises logprobs
        but returns none must not have signal S3 planned around it."""
        r = Registry.load()
        r.specs["stub"].capabilities = {Capability.LOGPROBS, Capability.SEED}
        r.apply_probe_results({"providers": {"stub": {"capabilities": ["seed"]}}})
        assert r.specs["stub"].capabilities == {Capability.SEED}

    def test_probe_can_disable_an_unreachable_provider(self):
        r = Registry.load()
        r.apply_probe_results({"providers": {"groq": {"reachable": False}}})
        assert r.specs["groq"].enabled is False


# ──────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────────────────────────────────────


class TestRateLimiter:
    def test_allows_requests_under_the_limit(self, tmp_path):
        rl = RateLimiter("t", rpm=100, state_dir=tmp_path, safety_margin=0.0)
        assert all(rl.acquire(timeout=1.0) for _ in range(5))

    def test_daily_cap_is_enforced(self, tmp_path):
        rl = RateLimiter("t", rpd=3, state_dir=tmp_path, safety_margin=0.0)
        assert sum(rl.acquire(timeout=0.5) for _ in range(5)) == 3
        assert rl.has_daily_capacity() is False

    def test_safety_margin_reserves_headroom(self, tmp_path):
        """Published limits are approximate and enforcement is often stricter."""
        rl = RateLimiter("t", rpd=100, state_dir=tmp_path, safety_margin=0.05)
        assert rl.rpd == 95

    def test_daily_count_survives_restart(self, tmp_path):
        """An overnight job that crashes and is relaunched must not reset the
        day's allowance and start hammering an exhausted quota."""
        first = RateLimiter("persist", rpd=10, state_dir=tmp_path, safety_margin=0.0)
        for _ in range(4):
            first.acquire(timeout=0.5)

        second = RateLimiter("persist", rpd=10, state_dir=tmp_path, safety_margin=0.0)
        assert second.snapshot().requests_today == 4
        assert second.snapshot().requests_remaining == 6

    def test_mark_exhausted_trusts_the_server(self, tmp_path):
        """Our counter can lag reality -- requests from another machine, or before
        this state file existed. When the server says done, we are done."""
        rl = RateLimiter("t", rpd=1000, state_dir=tmp_path, safety_margin=0.0)
        rl.acquire(timeout=0.5)
        assert rl.has_daily_capacity() is True
        rl.mark_exhausted()
        assert rl.has_daily_capacity() is False

    def test_no_limits_means_never_blocked(self, tmp_path):
        rl = RateLimiter("t", state_dir=tmp_path)
        assert all(rl.acquire(timeout=0.5) for _ in range(50))


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────


class _FailingProvider:
    """Provider that raises a chosen error, to drive failover paths."""

    def __init__(self, name: str, exc: Exception):
        self.name = name
        self._exc = exc
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        raise self._exc

    def capabilities(self, model):
        return {Capability.SYSTEM_ROLE}

    def available_models(self):
        return ["x"]

    def health_check(self):
        return True


class TestRouter:
    def test_generates_through_the_stub(self, tmp_path):
        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        assert router.generate(make_request("[F03] q")).text == FIXTURES_BY_ID["F03"].answer

    def test_second_identical_call_is_served_from_cache(self, tmp_path):
        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        req = make_request("[F01] q")
        assert router.generate(req).cached is False
        assert router.generate(req).cached is True
        assert router.stats.cache_hits == 1

    def test_records_usage_stats(self, tmp_path):
        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        router.generate(make_request("[F01] q"))
        assert router.stats.for_provider("stub").requests == 1

    @pytest.mark.parametrize(
        "exc",
        [
            RateLimitError("slow down", provider="a"),
            QuotaExhaustedError("done for today", provider="a"),
        ],
    )
    def test_fails_over_to_the_next_provider(self, tmp_path, exc):
        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        failing = _FailingProvider("failing", exc)
        # Splice a failing provider in front of the working stub.
        router._bound.insert(
            0,
            type(router._bound[0])(
                spec=router.registry.get("stub"),
                provider=failing,
                limiter=RateLimiter("failing", state_dir=tmp_path),
            ),
        )
        result = router.generate(make_request("[F03] q"))

        assert failing.calls == 1, "failing provider should have been tried"
        assert result.text == FIXTURES_BY_ID["F03"].answer, "should recover via the stub"

    def test_raises_when_every_provider_fails(self, tmp_path):
        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        router._bound = [
            type(router._bound[0])(
                spec=router.registry.get("stub"),
                provider=_FailingProvider("f", RateLimitError("nope", provider="f")),
                limiter=RateLimiter("f", state_dir=tmp_path),
            )
        ]
        with pytest.raises(ProviderError, match="all providers failed"):
            router.generate(make_request("[F03] q"))

    def test_empty_provider_list_gives_an_actionable_error(self):
        with pytest.raises(RuntimeError, match="check_env"):
            LLMRouter(providers=["nonexistent-provider"])

    def test_guaranteed_capabilities_is_the_intersection(self, tmp_path):
        """The honest basis for planning signal S3: what *every* live provider can
        do, not what any one of them can."""
        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        assert router.guaranteed_capabilities("stub-v1") <= router.capabilities("stub-v1")

    def test_quota_report_lists_every_provider(self, tmp_path):
        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        report = router.quota_report()
        assert len(report) == 1 and report[0]["provider"] == "stub"

    def test_satisfies_the_llm_provider_protocol(self, tmp_path):
        """The router must be a drop-in replacement for a single provider."""
        from pramana.generation.base import LLMProvider

        router = LLMRouter(providers=["stub"], cache=GenerationCache(tmp_path))
        assert isinstance(router, LLMProvider)

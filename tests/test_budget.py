import httpx
import pytest

from pramana.generation.base import GenerationRequest, ProviderError, TransientError
from pramana.generation.budget import provider_timeout, request_budget
from pramana.generation.providers.google_genai import GoogleGenAIProvider
from pramana.generation.providers.openai_compatible import OpenAICompatibleProvider


def test_deadline_rounding_cannot_expand_configured_timeout(monkeypatch):
    from pramana.generation import budget
    monkeypatch.setattr(budget, "monotonic", lambda: 0.0)
    with request_budget(1, 60):
        budget._active.get().deadline += 1e-12
        assert provider_timeout(120) == 60


@pytest.mark.parametrize("kind", ["openai", "google"])
def test_zero_retries_makes_one_call_and_budget_blocks_the_next(kind):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    provider = OpenAICompatibleProvider(name="test", base_url="http://test", api_key_env=None, max_retries=0) if kind == "openai" else GoogleGenAIProvider(api_key="test", max_retries=0)
    provider._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))

    def send():
        if kind == "openai":
            return provider._post_with_retries({}, GenerationRequest(messages=(), model="m"), "m")
        return provider._post_with_retries({}, "m")

    try:
        with request_budget(1, 60):
            assert send() == {"ok": True}
            with pytest.raises(ProviderError, match="budget exhausted"):
                send()
        assert len(calls) == 1
        assert calls[0].extensions["timeout"]["read"] <= 60
    finally:
        provider.close()


def test_budget_restores_context_and_expired_deadline_blocks_calls():
    with request_budget(2, 0), pytest.raises(ProviderError, match="budget exhausted"):
        provider_timeout(20)
    assert provider_timeout(20) == 20


def test_final_failed_attempt_does_not_sleep(monkeypatch):
    provider = OpenAICompatibleProvider(name="test", base_url="http://test", api_key_env=None, max_retries=0)
    provider._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    monkeypatch.setattr(provider, "_sleep_backoff", lambda *args: pytest.fail("No retries remain; do not sleep"))
    try:
        with pytest.raises(TransientError):
            provider._post_with_retries({}, GenerationRequest(messages=(), model="m"), "m")
    finally:
        provider.close()

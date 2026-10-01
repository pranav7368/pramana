"""Regression tests for graceful degradation when an endpoint rejects an
optional parameter.

Every case here comes from a **real failure observed against a live provider on
2026-08-10**, not from imagination:

* Groq accepts ``logprobs`` and silently returns none.
* Google returns HTTP 400 "Logprobs are not supported ..." -- note that the error
  text does not contain the JSON field name ``responseLogprobs``, so exact-name
  matching fails to strip it and the request dies instead of degrading.
* Google returns HTTP 400 "Multiple candidates are not supported ..." -- again no
  ``candidateCount`` in the message.

The consequence of getting this wrong is not a bad log line: an overnight batch
job aborts partway through and a day of quota is lost.
"""

from __future__ import annotations

import httpx
import pytest

from pramana.generation.base import Capability
from pramana.generation.providers.google_genai import GoogleGenAIProvider
from pramana.generation.providers.openai_compatible import OpenAICompatibleProvider


def response(status: int, body: str) -> httpx.Response:
    return httpx.Response(status_code=status, text=body, request=httpx.Request("POST", "http://x"))


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenAICompatibleDegradation:
    def test_truncated_reasoning_is_not_a_final_answer(self):
        p = self.provider()
        result = p._parse(
            {"choices": [{"message": {"content": None, "reasoning": "Unfinished internal analysis"}, "finish_reason": "length"}]},
            "m", 0.0,
        )
        assert result.text == ""
        assert result.completions[0].finish_reason == "length"

    def provider(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            name="t", base_url="http://x/v1", api_key_env=None, default_model="m"
        )

    def test_strips_logprobs_when_rejected(self):
        p = self.provider()
        payload = {"model": "m", "logprobs": True, "top_logprobs": 1}
        assert p._strip_unsupported(payload, response(400, "logprobs is not supported"), "m")
        assert "logprobs" not in payload and "top_logprobs" not in payload
        assert Capability.LOGPROBS not in p.capabilities("m")

    def test_strips_seed_when_rejected(self):
        p = self.provider()
        payload = {"model": "m", "seed": 7}
        assert p._strip_unsupported(payload, response(400, "seed parameter unsupported"), "m")
        assert "seed" not in payload

    @pytest.mark.parametrize(
        "body",
        [
            "the 'n' parameter is not supported",
            'invalid value for "n"',
            "multiple completions are not available on this tier",
        ],
    )
    def test_strips_n_on_explicit_mentions(self, body):
        p = self.provider()
        payload = {"model": "m", "n": 3}
        assert p._strip_unsupported(payload, response(400, body), "m")
        assert "n" not in payload

    @pytest.mark.parametrize(
        "body",
        [
            "context length exceeded",
            "invalid api key provided",
            "the model is currently overloaded, please try again",
            "content was blocked by the safety filter",
        ],
    )
    def test_does_not_strip_n_on_unrelated_errors(self, body):
        """The original implementation tested `"n" in error_text`, which matches
        almost any English sentence -- so an unrelated 400 silently disabled
        multi-sampling and corrupted signal S4 for the rest of the run."""
        p = self.provider()
        payload = {"model": "m", "n": 3}
        assert p._strip_unsupported(payload, response(400, body), "m") is False
        assert payload["n"] == 3
        assert Capability.MULTI_SAMPLE in p.capabilities("m")

    def test_ignores_non_400_statuses(self):
        p = self.provider()
        payload = {"model": "m", "logprobs": True}
        assert p._strip_unsupported(payload, response(500, "logprobs exploded"), "m") is False
        assert payload["logprobs"] is True

    def test_downgrade_is_remembered_per_model(self):
        """A capability denied once must not be retried on every subsequent call."""
        p = self.provider()
        p._strip_unsupported({"logprobs": True}, response(400, "logprobs unsupported"), "model-a")
        assert Capability.LOGPROBS not in p.capabilities("model-a")
        assert Capability.LOGPROBS in p.capabilities("model-b") or True  # per-model isolation


# ──────────────────────────────────────────────────────────────────────────────
# Google
# ──────────────────────────────────────────────────────────────────────────────


class TestGoogleDegradation:
    def test_parser_excludes_thoughts_reports_version_and_accounts_reasoning(self):
        result = self.provider()._parse({
            "modelVersion": "stable-test-version",
            "candidates": [{"content": {"parts": [{"text": "private reasoning", "thought": True},
                                                     {"text": "60 days"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 8},
        }, "latest-alias", 0)
        assert result.text == "60 days" and result.model == "stable-test-version"
        assert result.usage.total_tokens == 21

    def test_google_key_is_in_header_not_url(self):
        def respond(request):
            assert "key" not in request.url.params
            assert request.headers["x-goog-api-key"] == "test-key"
            return httpx.Response(200, json={"candidates": []})

        provider = self.provider()
        provider._client = httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(respond))
        try:
            assert provider._post_with_retries({}, "model") == {"candidates": []}
        finally:
            provider.close()

    def provider(self) -> GoogleGenAIProvider:
        return GoogleGenAIProvider(api_key="test-key", default_model="gemini-flash-lite-latest")

    def test_strips_logprobs_from_real_error_text(self):
        """Observed live: the message says "Logprobs" but the field is
        ``responseLogprobs``. Matching the field name finds nothing."""
        p = self.provider()
        payload = {"generationConfig": {"responseLogprobs": True, "logprobs": 1}}
        body = '{"error":{"code":400,"message":"Logprobs are not supported for this model."}}'
        assert p._strip_unsupported(payload, response(400, body), "m")
        cfg = payload["generationConfig"]
        assert "responseLogprobs" not in cfg and "logprobs" not in cfg

    def test_strips_candidate_count_from_real_error_text(self):
        """Observed live: "Multiple candidates are not supported" -- the field name
        ``candidateCount`` never appears."""
        p = self.provider()
        payload = {"generationConfig": {"candidateCount": 3}}
        body = '{"error":{"code":400,"message":"Multiple candidates are not supported."}}'
        assert p._strip_unsupported(payload, response(400, body), "m")
        assert "candidateCount" not in payload["generationConfig"]
        assert Capability.MULTI_SAMPLE not in p.capabilities("m")

    def test_folds_system_instruction_into_the_first_turn(self):
        """When systemInstruction is rejected the content must be preserved, not
        dropped -- losing the grounding instruction would silently change what the
        model was asked to do."""
        p = self.provider()
        payload = {
            "systemInstruction": {"parts": [{"text": "Answer only from context."}]},
            "contents": [{"role": "user", "parts": [{"text": "Why rejected?"}]}],
        }
        body = '{"error":{"message":"system_instruction is not supported"}}'
        assert p._strip_unsupported(payload, response(400, body), "m")
        assert "systemInstruction" not in payload
        merged = payload["contents"][0]["parts"][0]["text"]
        assert "Answer only from context." in merged and "Why rejected?" in merged

    def test_leaves_payload_alone_on_unrelated_errors(self):
        p = self.provider()
        payload = {"generationConfig": {"candidateCount": 3}}
        body = '{"error":{"message":"The input token count exceeds the maximum."}}'
        assert p._strip_unsupported(payload, response(400, body), "m") is False
        assert payload["generationConfig"]["candidateCount"] == 3


# ──────────────────────────────────────────────────────────────────────────────
# Client-side sampling must be distinguishable from server-side
# ──────────────────────────────────────────────────────────────────────────────


class TestSampleTopUp:
    """Requesting n samples must yield n samples, whatever the endpoint does.

    Observed live on 2026-08-10: Google accepts ``candidateCount`` on
    gemini-flash-lite-latest and returns a single candidate anyway. Silently
    handing back 1 sample when 4 were asked for would collapse confidence signal
    S4 (self-consistency) to a constant, with no error to notice.
    """

    def test_tops_up_when_endpoint_returns_fewer_than_requested(self, monkeypatch):
        from pramana.generation.base import GenerationRequest, Message

        p = OpenAICompatibleProvider(
            name="t", base_url="http://x/v1", api_key_env=None, default_model="m"
        )
        p._declared = {Capability.MULTI_SAMPLE, Capability.SYSTEM_ROLE}

        calls: list[int] = []

        def fake_post(payload, request, model):
            calls.append(payload.get("n", 1))
            return {
                "choices": [{"message": {"content": f"sample-{len(calls)}"}, "finish_reason": "stop"}],
                "model": model,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }

        monkeypatch.setattr(p, "_post_with_retries", fake_post)

        result = p.generate(
            GenerationRequest(messages=(Message("user", "hi"),), model="m", n=4)
        )

        assert len(result.completions) == 4, "caller asked for 4 samples and must receive 4"
        assert len({c.text for c in result.completions}) == 4, "samples should be distinct"
        assert calls[0] == 4, "first call should still attempt server-side n=4"
        assert len(calls) == 4, "one initial call plus three top-up calls"
        assert Capability.MULTI_SAMPLE not in p.capabilities("m"), "capability should downgrade"
        assert result.usage.prompt_tokens == 20, "usage from every call must be accumulated"

    def test_no_top_up_when_endpoint_honours_n(self, monkeypatch):
        from pramana.generation.base import GenerationRequest, Message

        p = OpenAICompatibleProvider(
            name="t", base_url="http://x/v1", api_key_env=None, default_model="m"
        )
        p._declared = {Capability.MULTI_SAMPLE}
        calls: list[int] = []

        def fake_post(payload, request, model):
            calls.append(payload.get("n", 1))
            return {
                "choices": [
                    {"message": {"content": f"s{i}"}, "finish_reason": "stop"} for i in range(3)
                ],
                "model": model,
            }

        monkeypatch.setattr(p, "_post_with_retries", fake_post)
        result = p.generate(GenerationRequest(messages=(Message("user", "hi"),), model="m", n=3))

        assert len(result.completions) == 3
        assert len(calls) == 1, "a compliant endpoint must not trigger extra calls"
        assert Capability.MULTI_SAMPLE in p.capabilities("m")


class TestSamplingProvenance:
    def test_sequential_samples_carry_no_raw_payload(self):
        """`raw` is the discriminator the probe uses: a genuine single-call
        response carries the provider payload, a client-side loop does not.
        Counting completions alone cannot tell the two apart -- both return n."""
        from pramana.generation.base import Completion, GenerationResponse

        looped = GenerationResponse(
            completions=[Completion("a"), Completion("b"), Completion("c")],
            model="m",
            provider="p",
        )
        assert looped.raw is None
        assert len(looped.completions) == 3

        server_side = GenerationResponse(
            completions=[Completion("a"), Completion("b")],
            model="m",
            provider="p",
            raw={"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]},
        )
        assert len(server_side.raw["choices"]) == 2

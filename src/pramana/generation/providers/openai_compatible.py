"""Adapter for any endpoint speaking the OpenAI Chat Completions dialect.

This single class covers the large majority of the ecosystem -- Groq, OpenRouter,
Cerebras, Together, DeepSeek, Mistral, Fireworks, xAI, Nebius, and every local
server that emulates the API (Ollama, vLLM, LM Studio, llama.cpp, TGI, LocalAI).

Deliberately implemented over ``httpx`` rather than the vendor SDKs: one HTTP
dependency instead of eight package installs that each pin their own transitive
versions, and no surprises when a vendor SDK diverges from the wire format.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Iterable
from typing import Any

import httpx

from pramana.generation.base import (
    AuthError,
    Capability,
    Completion,
    GenerationRequest,
    GenerationResponse,
    ModelNotAvailableError,
    ProviderError,
    QuotaExhaustedError,
    RateLimitError,
    TokenLogProb,
    TransientError,
    Usage,
)
from pramana.generation.budget import provider_timeout

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider:
    """Talks to any ``/chat/completions`` endpoint.

    Args:
        name: stable provider id, recorded in cache keys and manifests.
        base_url: API root, e.g. ``https://api.groq.com/openai/v1``.
        api_key_env: environment variable holding the key. Local servers that need
            no auth pass ``None``.
        default_model: used when a request does not name one.
        declared_capabilities: optimistic starting set, corrected by probing.
        timeout: per-request seconds.
        max_retries: attempts for *retryable* failures only.
        extra_headers: provider-specific headers (OpenRouter's attribution headers,
            for instance).
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key_env: str | None = None,
        default_model: str = "",
        declared_capabilities: set[Capability] | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        extra_headers: dict[str, str] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._extra_headers = extra_headers or {}
        self._declared = declared_capabilities or {
            Capability.SYSTEM_ROLE,
            Capability.MULTI_SAMPLE,
        }
        # Runtime downgrades: a capability that fails once is not retried.
        self._denied: dict[str, set[Capability]] = {}

        self._api_key = api_key if api_key is not None else (
            os.environ.get(api_key_env, "") if api_key_env else ""
        )
        self._api_key_env = api_key_env
        self._client: httpx.Client | None = None

    # ── plumbing ──────────────────────────────────────────────────────────────

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            headers = {"Content-Type": "application/json", **self._extra_headers}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.Client(
                base_url=self.base_url, headers=headers, timeout=self.timeout
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OpenAICompatibleProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_configured(self) -> bool:
        """False when an API key is required but absent -- the router skips such
        providers instead of burning a retry budget on guaranteed 401s."""
        return not self._api_key_env or bool(self._api_key)

    # ── capabilities ──────────────────────────────────────────────────────────

    def capabilities(self, model: str) -> set[Capability]:
        return self._declared - self._denied.get(model, set())

    def _deny(self, model: str, cap: Capability) -> None:
        self._denied.setdefault(model, set()).add(cap)
        log.info("provider=%s model=%s capability downgraded: %s", self.name, model, cap.value)

    # ── request construction ──────────────────────────────────────────────────

    def _build_payload(self, req: GenerationRequest) -> dict[str, Any]:
        model = req.model or self.default_model
        caps = self.capabilities(model)

        messages: list[dict[str, str]] = []
        if Capability.SYSTEM_ROLE in caps:
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
        else:
            # Fold system turns into the first user turn for endpoints that reject
            # a system role outright.
            system = "\n\n".join(m.content for m in req.messages if m.role == "system")
            for m in req.messages:
                if m.role == "system":
                    continue
                content = f"{system}\n\n{m.content}" if system and m.role == "user" else m.content
                messages.append({"role": m.role, "content": content})
                system = ""

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p,
        }
        if req.stop:
            payload["stop"] = list(req.stop)
        if req.n > 1 and Capability.MULTI_SAMPLE in caps:
            payload["n"] = req.n
        if req.seed is not None and Capability.SEED in caps:
            payload["seed"] = req.seed
        if req.want_logprobs and Capability.LOGPROBS in caps:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1
        payload.update(dict(req.extra))
        return payload

    # ── main entry point ──────────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        model = request.model or self.default_model
        if not model:
            raise ProviderError("no model specified and no default configured", provider=self.name)
        if not self.is_configured:
            raise AuthError(
                f"{self.name}: environment variable {self._api_key_env} is not set",
                provider=self.name,
            )

        # Loop client-side when the endpoint cannot sample n>1 in one call.
        if request.n > 1 and Capability.MULTI_SAMPLE not in self.capabilities(model):
            return self._generate_sequential_samples(request, model)

        started = time.perf_counter()
        data = self._post_with_retries(self._build_payload(request), request, model)
        latency_ms = (time.perf_counter() - started) * 1000
        response = self._parse(data, model, latency_ms)

        # An endpoint may accept `n` and return fewer completions anyway, or the
        # parameter may have been stripped mid-flight by a runtime downgrade.
        # Returning 1 sample when 4 were requested would silently reduce signal
        # S4 to a constant, so top up with sequential calls instead.
        if request.n > 1 and len(response.completions) < request.n:
            self._deny(model, Capability.MULTI_SAMPLE)
            return self._generate_sequential_samples(request, model, seed_with=response)
        return response

    def _generate_sequential_samples(
        self,
        request: GenerationRequest,
        model: str,
        seed_with: GenerationResponse | None = None,
    ) -> GenerationResponse:
        """Produce ``request.n`` completions one call at a time.

        ``seed_with`` carries completions already obtained, so a partial response
        is topped up rather than discarded and paid for twice.
        """
        merged: list[Completion] = list(seed_with.completions) if seed_with else []
        usage = Usage()
        if seed_with:
            usage.prompt_tokens = seed_with.usage.prompt_tokens
            usage.completion_tokens = seed_with.usage.completion_tokens

        started = time.perf_counter()
        for i in range(len(merged), request.n):
            single = request.with_overrides(
                n=1, seed=None if request.seed is None else request.seed + i
            )
            data = self._post_with_retries(self._build_payload(single), single, model)
            parsed = self._parse(data, model, 0.0)
            merged.extend(parsed.completions)
            usage.prompt_tokens += parsed.usage.prompt_tokens
            usage.completion_tokens += parsed.usage.completion_tokens

        return GenerationResponse(
            completions=merged,
            model=model,
            provider=self.name,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # ── transport with backoff ────────────────────────────────────────────────

    def _post_with_retries(
        self, payload: dict[str, Any], request: GenerationRequest, model: str
    ) -> dict[str, Any]:
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.post("/chat/completions", json=payload, timeout=provider_timeout(self.timeout))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = TransientError(f"{self.name}: {exc}", provider=self.name)
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                return resp.json()

            err = self._classify(resp, model, payload)

            # An unsupported optional parameter is a permanent, correctable fault:
            # downgrade the capability and retry once without it.
            if isinstance(err, ProviderError) and self._strip_unsupported(payload, resp, model):
                continue

            if not err.retryable:
                raise err

            last = err
            retry_after = getattr(err, "retry_after", None)
            if attempt < self.max_retries:
                self._sleep_backoff(attempt, retry_after)

        raise last or TransientError(f"{self.name}: exhausted retries", provider=self.name)

    def _classify(self, resp: httpx.Response, model: str, payload: dict[str, Any]) -> ProviderError:
        body = resp.text[:600]
        status = resp.status_code

        if status in (401, 403):
            return AuthError(f"{self.name}: auth rejected ({status}) {body}", provider=self.name)

        if status == 404:
            return ModelNotAvailableError(
                f"{self.name}: model '{model}' not found. {body}", provider=self.name
            )

        if status == 429:
            lowered = body.lower()
            # Distinguish "slow down" from "you are done for today": the first
            # deserves backoff, the second deserves failover to another provider.
            if any(k in lowered for k in ("daily", "per day", "quota", "credits", "exhausted")):
                return QuotaExhaustedError(
                    f"{self.name}: daily quota exhausted. {body}", provider=self.name
                )
            after = resp.headers.get("retry-after")
            return RateLimitError(
                f"{self.name}: rate limited. {body}",
                provider=self.name,
                retry_after=float(after) if after and after.replace(".", "", 1).isdigit() else None,
            )

        if status in _RETRYABLE_STATUS:
            return TransientError(f"{self.name}: HTTP {status}. {body}", provider=self.name)

        return ProviderError(f"{self.name}: HTTP {status}. {body}", provider=self.name)

    # Match on distinctive phrases from the error body, not on bare parameter
    # names. Naively testing `"n" in error_text` matches virtually every English
    # sentence and would strip multi-sampling on unrelated 400s.
    _UNSUPPORTED_HINTS: tuple[tuple[tuple[str, ...], str, Capability], ...] = (
        (("logprob",), "logprobs", Capability.LOGPROBS),
        (("seed",), "seed", Capability.SEED),
        (
            ("'n'", '"n"', "`n`", "parameter n", "multiple completions", "n>1", "n > 1"),
            "n",
            Capability.MULTI_SAMPLE,
        ),
    )

    def _strip_unsupported(
        self, payload: dict[str, Any], resp: httpx.Response, model: str
    ) -> bool:
        """Drop an optional parameter the endpoint rejected. Returns True if the
        request was modified and is worth retrying.

        Free-tier endpoints commonly accept ``logprobs`` in their documentation and
        reject it at runtime; treating that as fatal would lose a whole batch run.
        """
        if resp.status_code not in (400, 422):
            return False
        lowered = resp.text.lower()
        for hints, key, cap in self._UNSUPPORTED_HINTS:
            if any(h in lowered for h in hints) and key in payload:
                payload.pop(key, None)
                if key == "logprobs":
                    payload.pop("top_logprobs", None)
                self._deny(model, cap)
                return True
        return False

    def _sleep_backoff(self, attempt: int, retry_after: float | None = None) -> None:
        if retry_after is not None:
            delay = min(retry_after, 60.0)
        else:
            # Exponential with full jitter -- avoids synchronised retry storms when
            # several workers hit the same free-tier limit.
            delay = min(2.0**attempt, 30.0) * (0.5 + random.random() / 2)
        log.debug("provider=%s backing off %.1fs (attempt %d)", self.name, delay, attempt + 1)
        time.sleep(delay)

    # ── response parsing ──────────────────────────────────────────────────────

    def _parse(self, data: dict[str, Any], model: str, latency_ms: float) -> GenerationResponse:
        completions: list[Completion] = []

        for choice in data.get("choices", []):
            message = choice.get("message") or {}
            text = message.get("content") or ""
            # Internal reasoning is not a final answer, including when the
            # completion budget expires before content is produced.

            completions.append(
                Completion(
                    text=text,
                    finish_reason=choice.get("finish_reason"),
                    token_logprobs=self._parse_logprobs(choice.get("logprobs")),
                )
            )

        if not completions:
            raise ProviderError(
                f"{self.name}: response contained no choices: {str(data)[:300]}",
                provider=self.name,
            )

        raw_usage = data.get("usage") or {}
        return GenerationResponse(
            completions=completions,
            model=data.get("model", model),
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
            ),
            latency_ms=latency_ms,
            raw=data,
        )

    @staticmethod
    def _parse_logprobs(node: Any) -> list[TokenLogProb] | None:
        """Normalise the two logprob shapes seen in the wild.

        Modern (OpenAI-style): ``{"content": [{"token": t, "logprob": v}, ...]}``
        Legacy (completions):  ``{"tokens": [...], "token_logprobs": [...]}``
        """
        if not isinstance(node, dict):
            return None

        content = node.get("content")
        if isinstance(content, list) and content:
            out = [
                TokenLogProb(token=str(e.get("token", "")), logprob=float(e["logprob"]))
                for e in content
                if isinstance(e, dict) and e.get("logprob") is not None
            ]
            return out or None

        tokens, lps = node.get("tokens"), node.get("token_logprobs")
        if isinstance(tokens, list) and isinstance(lps, list):
            out = [
                TokenLogProb(token=str(t), logprob=float(v))
                # strict=False: a malformed provider payload with mismatched
                # lengths should yield fewer logprobs, not abort the whole run.
                for t, v in zip(tokens, lps, strict=False)
                if v is not None
            ]
            return out or None

        return None

    # ── introspection ─────────────────────────────────────────────────────────

    def available_models(self) -> Iterable[str]:
        try:
            resp = self.client.get("/models")
            if resp.status_code != 200:
                return []
            return sorted(
                str(m["id"]) for m in resp.json().get("data", []) if isinstance(m, dict) and "id" in m
            )
        except (httpx.TimeoutException, httpx.TransportError, ValueError):
            return []

    def health_check(self) -> bool:
        if not self.is_configured:
            return False
        try:
            resp = self.client.get("/models")
            return resp.status_code < 500
        except (httpx.TimeoutException, httpx.TransportError):
            return False

    def __repr__(self) -> str:
        return f"<OpenAICompatibleProvider {self.name} @ {self.base_url}>"

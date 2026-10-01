"""Native adapter for the Google Generative Language API (Gemini / Gemma).

Google's wire format differs from the OpenAI dialect -- ``contents`` instead of
``messages``, ``model`` instead of ``assistant``, config nested under
``generationConfig`` -- so it gets a native adapter rather than being forced
through the compatibility shim.

Included for two reasons: Google's free tier is one of the more generous, and its
models tend to be comparatively well-resourced for Hindi and Tamil, which matters
directly for this project.

This class is also the worked example for the question *"how do I connect an LLM
that is not OpenAI-compatible?"* -- implement ``LLMProvider``, normalise into
``GenerationResponse``, register it. Nothing else in the pipeline changes.
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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleGenAIProvider:
    """Adapter for ``models/{model}:generateContent``."""

    def __init__(
        self,
        *,
        name: str = "google",
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "GOOGLE_API_KEY",
        default_model: str = "gemini-2.0-flash",
        declared_capabilities: set[Capability] | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._declared = declared_capabilities or {
            Capability.SYSTEM_ROLE,
            Capability.MULTI_SAMPLE,
        }
        self._denied: dict[str, set[Capability]] = {}
        self._api_key_env = api_key_env
        self._api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self._client: httpx.Client | None = None

    # ── plumbing ──────────────────────────────────────────────────────────────

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> GoogleGenAIProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def capabilities(self, model: str) -> set[Capability]:
        return self._declared - self._denied.get(model, set())

    def _deny(self, model: str, cap: Capability) -> None:
        self._denied.setdefault(model, set()).add(cap)
        log.info("provider=%s model=%s capability downgraded: %s", self.name, model, cap.value)

    # ── request construction ──────────────────────────────────────────────────

    def _build_payload(self, req: GenerationRequest, model: str) -> dict[str, Any]:
        caps = self.capabilities(model)

        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for m in req.messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            # Gemini names the assistant turn "model".
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        gen_config: dict[str, Any] = {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_tokens,
            "topP": req.top_p,
        }
        if req.stop:
            gen_config["stopSequences"] = list(req.stop)
        if req.n > 1 and Capability.MULTI_SAMPLE in caps:
            gen_config["candidateCount"] = req.n
        if req.seed is not None and Capability.SEED in caps:
            gen_config["seed"] = req.seed
        if req.want_logprobs and Capability.LOGPROBS in caps:
            gen_config["responseLogprobs"] = True
            gen_config["logprobs"] = 1

        payload: dict[str, Any] = {"contents": contents, "generationConfig": gen_config}

        if system_parts:
            text = "\n\n".join(system_parts)
            if Capability.SYSTEM_ROLE in caps:
                payload["systemInstruction"] = {"parts": [{"text": text}]}
            elif contents:
                first = contents[0]["parts"][0]
                first["text"] = f"{text}\n\n{first['text']}"

        payload.update(dict(req.extra))
        return payload

    # ── main entry point ──────────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        model = (request.model or self.default_model).removeprefix("models/")
        if not model:
            raise ProviderError("no model specified and no default configured", provider=self.name)
        if not self.is_configured:
            raise AuthError(
                f"{self.name}: environment variable {self._api_key_env} is not set",
                provider=self.name,
            )

        if request.n > 1 and Capability.MULTI_SAMPLE not in self.capabilities(model):
            return self._generate_sequential_samples(request, model)

        started = time.perf_counter()
        data = self._post_with_retries(self._build_payload(request, model), model)
        response = self._parse(data, model, (time.perf_counter() - started) * 1000)

        # Google silently ignores candidateCount on several free-tier models, and
        # a runtime downgrade may have stripped it. Returning 1 sample when 4 were
        # requested would reduce signal S4 to a constant, so top up sequentially.
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
            data = self._post_with_retries(self._build_payload(single, model), model)
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

    # ── transport ─────────────────────────────────────────────────────────────

    def _post_with_retries(self, payload: dict[str, Any], model: str) -> dict[str, Any]:
        url = f"/models/{model}:generateContent"
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.post(url, headers={"x-goog-api-key": self._api_key}, json=payload, timeout=provider_timeout(self.timeout))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = TransientError(f"{self.name}: {exc}", provider=self.name)
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 400 and self._strip_unsupported(payload, resp, model):
                continue

            err = self._classify(resp, model)
            if not err.retryable:
                raise err
            last = err
            if attempt < self.max_retries:
                self._sleep_backoff(attempt, getattr(err, "retry_after", None))

        raise last or TransientError(f"{self.name}: exhausted retries", provider=self.name)

    def _classify(self, resp: httpx.Response, model: str) -> ProviderError:
        body = resp.text[:600]
        status = resp.status_code

        if status in (401, 403):
            return AuthError(f"{self.name}: auth rejected ({status}). {body}", provider=self.name)
        if status == 404:
            return ModelNotAvailableError(
                f"{self.name}: model '{model}' not found. {body}", provider=self.name
            )
        if status == 429:
            lowered = body.lower()
            if any(k in lowered for k in ("per day", "daily", "quota", "exhausted")):
                return QuotaExhaustedError(
                    f"{self.name}: daily quota exhausted. {body}", provider=self.name
                )
            after = resp.headers.get("retry-after")
            return RateLimitError(
                f"{self.name}: rate limited. {body}",
                provider=self.name,
                retry_after=float(after) if after and after.replace(".", "", 1).isdigit() else None,
            )
        if status >= 500:
            return TransientError(f"{self.name}: HTTP {status}. {body}", provider=self.name)
        return ProviderError(f"{self.name}: HTTP {status}. {body}", provider=self.name)

    # Match on what the error *says*, not on the parameter name. Google reports
    # "Logprobs are not supported" and "Multiple candidates are not supported" --
    # neither string contains the JSON field name, so exact-name matching silently
    # fails to strip and the request dies instead of degrading.
    _UNSUPPORTED_HINTS: tuple[tuple[tuple[str, ...], str, Capability], ...] = (
        (("logprob",), "responseLogprobs", Capability.LOGPROBS),
        (("candidatecount", "multiple candidates", "candidate_count"), "candidateCount", Capability.MULTI_SAMPLE),
        (("seed",), "seed", Capability.SEED),
    )

    def _strip_unsupported(self, payload: dict[str, Any], resp: httpx.Response, model: str) -> bool:
        cfg = payload.get("generationConfig", {})
        lowered = resp.text.lower()

        for hints, key, cap in self._UNSUPPORTED_HINTS:
            if any(h in lowered for h in hints) and key in cfg:
                cfg.pop(key, None)
                if key == "responseLogprobs":
                    cfg.pop("logprobs", None)
                self._deny(model, cap)
                return True

        if "system" in lowered and "systemInstruction" in payload:
            text = payload.pop("systemInstruction")["parts"][0]["text"]
            if payload.get("contents"):
                first = payload["contents"][0]["parts"][0]
                first["text"] = f"{text}\n\n{first['text']}"
            self._deny(model, Capability.SYSTEM_ROLE)
            return True
        return False

    def _sleep_backoff(self, attempt: int, retry_after: float | None = None) -> None:
        delay = (
            min(retry_after, 60.0)
            if retry_after is not None
            else min(2.0**attempt, 30.0) * (0.5 + random.random() / 2)
        )
        time.sleep(delay)

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse(self, data: dict[str, Any], model: str, latency_ms: float) -> GenerationResponse:
        completions: list[Completion] = []

        for cand in data.get("candidates", []):
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
            completions.append(
                Completion(
                    text=text,
                    finish_reason=cand.get("finishReason"),
                    token_logprobs=self._parse_logprobs(cand.get("logprobsResult")),
                )
            )

        if not completions:
            # A safety block returns no candidates -- surface the reason rather
            # than a bare "empty response".
            feedback = data.get("promptFeedback") or {}
            raise ProviderError(
                f"{self.name}: no candidates returned "
                f"(blockReason={feedback.get('blockReason', 'unknown')})",
                provider=self.name,
            )

        meta = data.get("usageMetadata") or {}
        return GenerationResponse(
            completions=completions,
            model=data.get("modelVersion") or model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(meta.get("promptTokenCount", 0) or 0),
                completion_tokens=int(meta.get("candidatesTokenCount", 0) or 0) + int(meta.get("thoughtsTokenCount", 0) or 0),
            ),
            latency_ms=latency_ms,
            raw=data,
        )

    @staticmethod
    def _parse_logprobs(node: Any) -> list[TokenLogProb] | None:
        if not isinstance(node, dict):
            return None
        chosen = node.get("chosenCandidates")
        if not isinstance(chosen, list) or not chosen:
            return None
        out = [
            TokenLogProb(token=str(e.get("token", "")), logprob=float(e["logProbability"]))
            for e in chosen
            if isinstance(e, dict) and e.get("logProbability") is not None
        ]
        return out or None

    # ── introspection ─────────────────────────────────────────────────────────

    def available_models(self) -> Iterable[str]:
        try:
            resp = self.client.get("/models", params={"key": self._api_key})
            if resp.status_code != 200:
                return []
            return sorted(
                str(m["name"]).removeprefix("models/")
                for m in resp.json().get("models", [])
                if isinstance(m, dict) and "name" in m
            )
        except (httpx.TimeoutException, httpx.TransportError, ValueError):
            return []

    def health_check(self) -> bool:
        if not self.is_configured:
            return False
        try:
            return self.client.get("/models", params={"key": self._api_key}).status_code < 500
        except (httpx.TimeoutException, httpx.TransportError):
            return False

    def __repr__(self) -> str:
        return f"<GoogleGenAIProvider {self.name} @ {self.base_url}>"

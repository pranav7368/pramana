"""Core generation contracts.

Everything downstream of this module depends only on the ``LLMProvider`` protocol
and the request/response dataclasses defined here.  No provider SDK is imported
at this level, which is what allows an arbitrary LLM to be plugged in without
touching the pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Capability(StrEnum):
    """Optional provider features the pipeline can exploit when present.

    Capabilities are *discovered by probing*, never assumed from documentation --
    free-tier endpoints frequently advertise features they do not deliver.
    See ``scripts/probe_providers.py``.
    """

    LOGPROBS = "logprobs"
    """Returns per-token log probabilities (confidence signal S3)."""

    SEED = "seed"
    """Honours a deterministic seed parameter."""

    SYSTEM_ROLE = "system_role"
    """Accepts a distinct system message rather than folding it into the user turn."""

    JSON_MODE = "json_mode"
    """Can be constrained to emit syntactically valid JSON."""

    STREAMING = "streaming"
    MULTI_SAMPLE = "multi_sample"
    """Supports n>1 in a single request (otherwise samples are looped client-side)."""


@dataclass(frozen=True, slots=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A single generation call, fully specified.

    Frozen and hashable-by-content so it can key the response cache.  Two requests
    that differ in any field that could change the output must produce different
    cache keys -- see ``pramana.generation.cache.request_fingerprint``.
    """

    messages: tuple[Message, ...]
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    n: int = 1
    seed: int | None = None
    stop: tuple[str, ...] = ()
    want_logprobs: bool = False
    extra: tuple[tuple[str, Any], ...] = ()
    """Provider-specific passthrough params, as a sorted tuple for hashability."""

    def with_overrides(self, **kwargs: Any) -> GenerationRequest:
        from dataclasses import replace

        return replace(self, **kwargs)


@dataclass(slots=True)
class TokenLogProb:
    token: str
    logprob: float


@dataclass(slots=True)
class Completion:
    """One sampled continuation."""

    text: str
    finish_reason: str | None = None
    token_logprobs: list[TokenLogProb] | None = None

    @property
    def mean_logprob(self) -> float | None:
        """Mean token log-probability -- confidence signal S3.

        ``None`` when the provider does not return logprobs, which the confidence
        fusion model treats as a *missing feature* (refit without S3) rather than
        imputing a value.  Imputation would silently corrupt calibration.
        """
        if not self.token_logprobs:
            return None
        return sum(t.logprob for t in self.token_logprobs) / len(self.token_logprobs)


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class GenerationResponse:
    """Provider-neutral result. Every adapter normalises into this shape."""

    completions: list[Completion]
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cached: bool = False
    created_at: float = field(default_factory=time.time)
    raw: dict[str, Any] | None = None
    """Untouched provider payload, retained for debugging and audit."""

    @property
    def text(self) -> str:
        """First completion's text -- the common single-sample case."""
        if not self.completions:
            raise ValueError("GenerationResponse contains no completions")
        return self.completions[0].text

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.completions]

    @property
    def has_logprobs(self) -> bool:
        return any(c.token_logprobs for c in self.completions)


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────


class ProviderError(RuntimeError):
    """Base class for all provider failures."""

    def __init__(self, message: str, *, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class InvalidResponseError(ProviderError):
    """The provider responded, but did not return a complete usable answer."""


class RateLimitError(ProviderError):
    """HTTP 429 or a locally enforced limit. Always retryable."""

    def __init__(self, message: str, *, provider: str = "", retry_after: float | None = None):
        super().__init__(message, provider=provider, retryable=True)
        self.retry_after = retry_after


class LocalCapacityError(RateLimitError):
    """Declared client-side capacity reached, not an upstream 429 response."""


class QuotaExhaustedError(ProviderError):
    """Daily/monthly allowance spent. Not retryable *today* -- the router fails
    over to another provider rather than backing off."""

    def __init__(self, message: str, *, provider: str = "", resets_at: float | None = None):
        super().__init__(message, provider=provider, retryable=False)
        self.resets_at = resets_at


class AuthError(ProviderError):
    """Missing or rejected credentials. Never retryable; surfaces immediately so a
    misconfigured key is not mistaken for a transient outage."""


class ModelNotAvailableError(ProviderError):
    """The requested model is unknown to this provider. Router tries the next one."""


class TransientError(ProviderError):
    """Timeout, connection reset, 5xx. Retryable with backoff."""

    def __init__(self, message: str, *, provider: str = ""):
        super().__init__(message, provider=provider, retryable=True)


# ──────────────────────────────────────────────────────────────────────────────
# The plug-in contract
# ──────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class LLMProvider(Protocol):
    """The single interface every backend implements.

    Implementing this protocol is *all* that is required to connect a new LLM --
    hosted API, local server, or an offline stub.
    """

    name: str
    """Stable identifier, e.g. "groq". Appears in cache keys and run manifests."""

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    def capabilities(self, model: str) -> set[Capability]:
        """Features believed available for ``model``.

        Populated by probing where possible; a declared capability that fails at
        runtime is downgraded rather than raising.
        """
        ...

    def available_models(self) -> Iterable[str]: ...

    def health_check(self) -> bool:
        """Cheap liveness probe. Used by the router to skip dead providers."""
        ...

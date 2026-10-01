"""Provider-agnostic LLM generation layer.

Typical use -- the pipeline never names a provider:

    from pramana.generation import LLMRouter, GenerationRequest, Message

    router = LLMRouter()                       # reads config/providers.yaml + .env
    response = router.generate(GenerationRequest(
        messages=(Message("system", "Answer only from the context."),
                  Message("user", "Why was my claim rejected?")),
        model="llama3",                        # alias resolved per provider
        n=4,                                   # self-consistency samples (signal S4)
        want_logprobs=True,                    # dropped silently if unsupported
    ))

Offline, with no key and no network:

    router = LLMRouter(providers=["stub"])

Connecting a new LLM: add an entry to ``config/providers.yaml``. Only an endpoint
that does not speak the OpenAI dialect needs new code -- implement ``LLMProvider``
and register it with ``@register_adapter``.
"""

from pramana.generation.base import (
    AuthError,
    Capability,
    Completion,
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    Message,
    ModelNotAvailableError,
    ProviderError,
    QuotaExhaustedError,
    RateLimitError,
    TokenLogProb,
    TransientError,
    Usage,
)
from pramana.generation.cache import GenerationCache, request_fingerprint
from pramana.generation.ratelimit import RateLimiter
from pramana.generation.registry import (
    ProviderSpec,
    Registry,
    load_registry,
    register_adapter,
)
from pramana.generation.router import LLMRouter
from pramana.generation.stub import FIXTURES, Fixture, StubProvider, fixtures_for

__all__ = [
    "FIXTURES",
    "AuthError",
    "Capability",
    "Completion",
    "Fixture",
    "GenerationCache",
    "GenerationRequest",
    "GenerationResponse",
    "LLMProvider",
    "LLMRouter",
    "Message",
    "ModelNotAvailableError",
    "ProviderError",
    "ProviderSpec",
    "QuotaExhaustedError",
    "RateLimitError",
    "RateLimiter",
    "Registry",
    "StubProvider",
    "TokenLogProb",
    "TransientError",
    "Usage",
    "fixtures_for",
    "load_registry",
    "register_adapter",
    "request_fingerprint",
]

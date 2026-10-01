"""Multi-provider router — the component that turns several small free tiers
into one usable generation budget.

Responsibilities, in the order they execute:

1. **Cache lookup.** A request already answered never reaches the network.
2. **Client-side throttling.** Stay under declared limits instead of being 429'd.
3. **Provider selection.** Priority order, round-robin, or pinned.
4. **Failover.** On rate limit, quota exhaustion, or an unavailable model, move to
   the next provider rather than failing the run.
5. **Accounting.** Record usage, cache the result, keep per-provider stats.

The practical effect: a 5,200-call generation job can be launched overnight and
will finish across whatever quota happens to be available, resuming cleanly if
it is interrupted.
"""

from __future__ import annotations

import itertools
import logging
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from pramana.generation.base import (
    AuthError,
    Capability,
    GenerationRequest,
    GenerationResponse,
    InvalidResponseError,
    LLMProvider,
    LocalCapacityError,
    ModelNotAvailableError,
    ProviderError,
    QuotaExhaustedError,
    RateLimitError,
    TransientError,
)
from pramana.generation.budget import remaining_seconds
from pramana.generation.cache import GenerationCache, request_fingerprint
from pramana.generation.ratelimit import RateLimiter
from pramana.generation.registry import ProviderSpec, Registry, load_registry

log = logging.getLogger(__name__)


def _usable_response(response: GenerationResponse) -> bool:
    """Reasoning-only, blocked and truncated text must never be accepted/cached."""
    return bool(response.completions) and all(
        isinstance(c.text, str) and c.text.strip() and (c.finish_reason or "").lower() not in {
            "length", "max_tokens", "content_filter", "safety", "recitation",
        }
        for c in response.completions
    )

_FAILOVER_ERRORS: dict[str, type[ProviderError]] = {
    "rate_limit": RateLimitError,
    "quota_exhausted": QuotaExhaustedError,
    "model_not_available": ModelNotAvailableError,
    "transient": TransientError,
}


@dataclass(slots=True)
class ProviderStats:
    requests: int = 0
    failures: int = 0
    failovers_away: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms_total: float = 0.0
    error_types: dict[str, int] = field(default_factory=dict)
    resolved_models: list[str] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        return self.latency_ms_total / self.requests if self.requests else 0.0


@dataclass(slots=True)
class RouterStats:
    per_provider: dict[str, ProviderStats] = field(default_factory=dict)
    cache_hits: int = 0
    total_calls: int = 0

    def for_provider(self, name: str) -> ProviderStats:
        return self.per_provider.setdefault(name, ProviderStats())

    def summary(self) -> str:
        lines = [
            f"calls={self.total_calls} cache_hits={self.cache_hits} "
            f"({self.cache_hits / self.total_calls:.1%})"
            if self.total_calls
            else "no calls"
        ]
        for name, s in sorted(self.per_provider.items()):
            lines.append(
                f"  {name:<16} req={s.requests:<5} fail={s.failures:<3} "
                f"failover={s.failovers_away:<3} tok={s.prompt_tokens + s.completion_tokens:<8} "
                f"lat={s.mean_latency_ms:.0f}ms"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class _Bound:
    """A provider instance with its spec and limiter."""

    spec: ProviderSpec
    provider: LLMProvider
    limiter: RateLimiter


class LLMRouter:
    """Routes ``GenerationRequest``s across providers with caching and failover.

    Satisfies the ``LLMProvider`` protocol itself, so it is a drop-in replacement
    for a single provider anywhere in the pipeline.
    """

    name = "router"

    def __init__(
        self,
        *,
        registry: Registry | None = None,
        providers: Sequence[str] | None = None,
        cache: GenerationCache | None = None,
        enforce_limits: bool | None = None,
        strategy: str | None = None,
        rate_limit_wait_s: float = 300.0,
    ) -> None:
        self.registry = registry or load_registry()
        policy = self.registry.policy

        self.strategy = strategy or policy.strategy
        self.enforce_limits = (
            policy.respect_declared_limits if enforce_limits is None else enforce_limits
        )
        self.max_failovers = policy.max_failovers
        self.failover_types = tuple(
            _FAILOVER_ERRORS[k] for k in policy.failover_on if k in _FAILOVER_ERRORS
        )
        self.cache = cache or GenerationCache(enabled=policy.cache_enabled)
        self.stats = RouterStats()
        self.rate_limit_wait_s = rate_limit_wait_s

        names = list(providers) if providers else [s.name for s in self.registry.usable()]
        if policy.pinned_provider and not providers:
            names = [policy.pinned_provider]

        self._bound: list[_Bound] = []
        for name in names:
            try:
                spec = self.registry.get(name)
                self._bound.append(
                    _Bound(
                        spec=spec,
                        provider=self.registry.build(name),
                        limiter=RateLimiter(
                            name,
                            rpm=spec.limits.rpm,
                            tpm=spec.limits.tpm,
                            rpd=spec.limits.rpd,
                            tpd=spec.limits.tpd,
                        ),
                    )
                )
            except Exception as exc:
                log.warning("router: skipping provider '%s': %s", name, type(exc).__name__)  # kill the rest

        if not self._bound:
            raise RuntimeError(
                "no usable LLM providers. Set at least one API key (e.g. GROQ_API_KEY) "
                "in .env, or use providers=['stub'] for offline development. "
                "Run `python scripts/check_env.py` to diagnose."
            )

        self._rr: Iterator[int] = itertools.cycle(range(len(self._bound)))
        log.info("router ready with providers: %s", [b.spec.name for b in self._bound])

    # ── selection ─────────────────────────────────────────────────────────────

    def _candidates(self) -> list[_Bound]:
        """Providers to try, best first, excluding those out of daily quota."""
        live = [
            b
            for b in self._bound
            if b.spec.enabled and (not self.enforce_limits or b.limiter.has_daily_capacity())
        ]
        if not live:
            # Everything is exhausted; return all so the caller gets a real
            # provider error rather than a silent empty list.
            return [b for b in self._bound if b.spec.enabled]

        if self.strategy == "round_robin":
            start = next(self._rr) % len(live)
            return live[start:] + live[:start]
        return live  # already priority-ordered by Registry.usable()

    # ── main entry point ──────────────────────────────────────────────────────

    def generate_validated(self, request: GenerationRequest, validator: Callable[[str], bool]) -> GenerationResponse:
        """Fail over when a stage gets usable text in an invalid required format."""
        return self.generate(request, validator=lambda r: all(validator(c.text) for c in r.completions))

    def generate(self, request: GenerationRequest, *, validator: Callable[[GenerationResponse], bool] | None = None) -> GenerationResponse:
        self.stats.total_calls += 1
        candidates = self._candidates()
        errors: list[str] = []

        for attempt, bound in enumerate(candidates):
            if attempt > self.max_failovers:
                break

            spec, provider, limiter = bound.spec, bound.provider, bound.limiter
            model = spec.resolve_model(request.model) if request.model else spec.default_model
            if not model:
                errors.append(f"{spec.name}: no model resolved")
                continue

            # Drop logprobs for providers known not to support them, so the
            # request is not rejected outright for an optional feature.
            local = request.with_overrides(
                model=model, extra=tuple((spec.request_defaults | dict(request.extra)).items()),
            )
            if local.want_logprobs and Capability.LOGPROBS not in provider.capabilities(model):
                local = local.with_overrides(want_logprobs=False)

            key = request_fingerprint(local, spec.name, model)
            if (hit := self.cache.get(key)) is not None and _usable_response(hit) and (validator is None or validator(hit)):
                self.stats.cache_hits += 1
                return hit

            reservation = uuid.uuid4().hex if self.enforce_limits else None
            # Include prompt bytes as a conservative estimate, not just output.
            estimate = local.max_tokens * local.n + sum(len(m.content.encode("utf-8")) for m in local.messages) // 3
            if self.enforce_limits and not limiter.acquire(tokens=estimate, reservation=reservation, timeout=remaining_seconds(self.rate_limit_wait_s)):
                log.info("router: %s has no capacity, failing over", spec.name)
                self._note_failover(spec.name, LocalCapacityError("Local rate/quota capacity", provider=spec.name), errors)
                continue

            try:
                response = provider.generate(local)
            except QuotaExhaustedError as exc:
                if reservation:
                    limiter.retain_reservation(reservation)
                limiter.mark_exhausted()  # trust the server over our counter
                self._note_failover(spec.name, exc, errors)
                continue
            except self.failover_types as exc:
                if reservation:
                    limiter.retain_reservation(reservation)
                self._note_failover(spec.name, exc, errors)
                continue
            except AuthError as exc:
                if reservation:
                    limiter.retain_reservation(reservation)
                # A bad key is a configuration bug, not a transient fault. Skip
                # the provider for this run but keep going.
                log.error("router: %s auth failed, disabling for this run: %s", spec.name, type(exc).__name__)
                bound.spec.enabled = False
                self._note_failover(spec.name, exc, errors)
                continue
            except ProviderError as exc:
                if reservation:
                    limiter.retain_reservation(reservation)
                self._note_failover(spec.name, exc, errors)
                continue
            except (ValueError, TypeError, KeyError, AttributeError):
                if reservation:
                    limiter.retain_reservation(reservation)
                self._note_failover(spec.name, InvalidResponseError("Malformed provider payload", provider=spec.name), errors)
                continue

            self._record(spec.name, response, reservation=reservation)
            if not _usable_response(response) or (validator is not None and not validator(response)):
                self._note_failover(spec.name, InvalidResponseError(
                    "Empty, blocked, or truncated completion", provider=spec.name,
                ), errors)
                continue
            self.cache.put(key, response)
            return response

        raise ProviderError(
            "all providers failed for this request:\n  " + "\n  ".join(errors or ["<none tried>"]),
            provider="router",
        )

    def _note_failover(self, name: str, exc: Exception, errors: list[str]) -> None:
        s = self.stats.for_provider(name)
        s.failures += 1
        s.failovers_away += 1
        kind = type(exc).__name__
        s.error_types[kind] = s.error_types.get(kind, 0) + 1
        log.info("router: failing over from %s (%s)", name, type(exc).__name__)
        errors.append(f"{name}: {type(exc).__name__}")

    def _record(self, name: str, response: GenerationResponse, *, reservation: str | None = None) -> None:
        s = self.stats.for_provider(name)
        if response.model not in s.resolved_models:
            s.resolved_models.append(response.model)
        s.requests += 1
        s.prompt_tokens += response.usage.prompt_tokens
        s.completion_tokens += response.usage.completion_tokens
        s.latency_ms_total += response.latency_ms
        for b in self._bound:
            if b.spec.name == name:
                b.limiter.record_usage(response.usage.prompt_tokens + response.usage.completion_tokens, reservation=reservation)
                break

    # ── LLMProvider protocol surface ──────────────────────────────────────────

    def capabilities(self, model: str) -> set[Capability]:
        """Union across live providers — what *some* backend can do."""
        caps: set[Capability] = set()
        for b in self._bound:
            caps |= b.provider.capabilities(b.spec.resolve_model(model) if model else model)
        return caps

    def guaranteed_capabilities(self, model: str = "") -> set[Capability]:
        """Intersection — what *every* live provider can do.

        The honest basis for planning a confidence signal: if logprobs are not in
        this set, signal S3 cannot be relied upon across the run, and the fusion
        model is fitted without it.
        """
        sets = [
            b.provider.capabilities(b.spec.resolve_model(model) if model else model)
            for b in self._bound
        ]
        return set.intersection(*sets) if sets else set()

    def available_models(self) -> list[str]:
        out: list[str] = []
        for b in self._bound:
            out.extend(f"{b.spec.name}/{m}" for m in b.provider.available_models())
        return out

    def health_check(self) -> bool:
        return any(b.provider.health_check() for b in self._bound)

    # ── reporting ─────────────────────────────────────────────────────────────

    def quota_report(self) -> list[dict[str, object]]:
        return [
            {
                "provider": b.spec.name,
                "priority": b.spec.priority,
                "requests_today": (snap := b.limiter.snapshot()).requests_today,
                "requests_remaining": snap.requests_remaining,
                "exhausted": snap.exhausted,
            }
            for b in self._bound
        ]

    def close(self) -> None:
        for b in self._bound:
            if hasattr(b.provider, "close"):
                b.provider.close()  # type: ignore[attr-defined]

    def __enter__(self) -> LLMRouter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<LLMRouter providers={[b.spec.name for b in self._bound]} strategy={self.strategy}>"

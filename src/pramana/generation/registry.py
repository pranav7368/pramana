"""Provider registry — builds ``LLMProvider`` instances from declarative config.

Connecting a new LLM is a two-line YAML edit when the endpoint speaks the OpenAI
dialect, and a new adapter class plus one ``@register_adapter`` decorator when it
does not.  Nothing in the pipeline is touched either way.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pramana.generation.base import Capability, LLMProvider

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_REGISTRY_PATH = CONFIG_DIR / "providers.yaml"

# ──────────────────────────────────────────────────────────────────────────────
# Adapter plug-in table
# ──────────────────────────────────────────────────────────────────────────────

AdapterFactory = Callable[..., LLMProvider]
_ADAPTERS: dict[str, AdapterFactory] = {}


def register_adapter(name: str) -> Callable[[AdapterFactory], AdapterFactory]:
    """Register an adapter factory under ``name``, matching ``adapter:`` in YAML.

    >>> @register_adapter("my_llm")
    ... def _build(**kw): return MyProvider(**kw)
    """

    def deco(factory: AdapterFactory) -> AdapterFactory:
        _ADAPTERS[name] = factory
        return factory

    return deco


def _lazy(module: str, cls: str) -> AdapterFactory:
    """Import an adapter only when it is actually instantiated.

    Keeps a missing optional dependency from breaking the whole registry: if
    ``httpx`` is absent, the stub adapter still works and offline development
    continues.
    """

    def factory(**kwargs: Any) -> LLMProvider:
        import importlib

        return getattr(importlib.import_module(module), cls)(**kwargs)

    return factory


_ADAPTERS["openai_compatible"] = _lazy(
    "pramana.generation.providers.openai_compatible", "OpenAICompatibleProvider"
)
_ADAPTERS["google_genai"] = _lazy(
    "pramana.generation.providers.google_genai", "GoogleGenAIProvider"
)
_ADAPTERS["stub"] = _lazy("pramana.generation.stub", "StubProvider")


# ──────────────────────────────────────────────────────────────────────────────
# Config objects
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Limits:
    """Documented free-tier limits, enforced client-side.

    Throttling ourselves is strictly better than being throttled: a 429 costs a
    round trip and a backoff, and repeated 429s can get a key blocked.
    """

    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Limits:
        d = d or {}
        return cls(rpm=d.get("rpm"), rpd=d.get("rpd"), tpm=d.get("tpm"), tpd=d.get("tpd"))


@dataclass(slots=True)
class ProviderSpec:
    name: str
    adapter: str
    default_model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    enabled: bool = True
    priority: int = 50
    capabilities: set[Capability] = field(default_factory=set)
    limits: Limits = field(default_factory=Limits)
    models: dict[str, str] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    request_defaults: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timeout_s: float = 120.0
    max_retries: int = 4

    @property
    def is_configured(self) -> bool:
        """True when credentials are present, or none are needed (local servers)."""
        return not self.api_key_env or bool(os.environ.get(self.api_key_env))

    def resolve_model(self, alias_or_id: str) -> str:
        """Map a short alias to a concrete model id, passing through unknown ids.

        Aliases let an experiment config say ``llama3`` and stay valid when the
        provider renames the underlying checkpoint.
        """
        return self.models.get(alias_or_id, alias_or_id) or self.default_model


@dataclass(slots=True)
class RouterPolicy:
    strategy: str = "priority_with_failover"
    pinned_provider: str | None = None
    cache_enabled: bool = True
    respect_declared_limits: bool = True
    failover_on: tuple[str, ...] = (
        "rate_limit",
        "quota_exhausted",
        "model_not_available",
        "transient",
    )
    max_failovers: int = 3


@dataclass(slots=True)
class Registry:
    specs: dict[str, ProviderSpec]
    policy: RouterPolicy
    source: Path | None = None

    # ── queries ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> ProviderSpec:
        try:
            return self.specs[name]
        except KeyError:
            raise KeyError(
                f"unknown provider '{name}'. Known: {sorted(self.specs)}"
            ) from None

    def usable(self) -> list[ProviderSpec]:
        """Enabled providers that actually have credentials, best priority first.

        Filtering on credentials here means the router never wastes a failover
        slot on a provider whose key was never set.
        """
        return sorted(
            (s for s in self.specs.values() if s.enabled and s.is_configured),
            key=lambda s: (s.priority, s.name),
        )

    def diagnose(self) -> list[tuple[str, str]]:
        """(provider, status) for every entry — the basis of ``check_env``."""
        out: list[tuple[str, str]] = []
        for spec in sorted(self.specs.values(), key=lambda s: (s.priority, s.name)):
            if not spec.enabled:
                status = "disabled"
            elif not spec.api_key_env:
                status = "ready (no key required)"
            elif spec.is_configured:
                status = f"ready ({spec.api_key_env} set)"
            else:
                status = f"MISSING KEY ({spec.api_key_env})"
            out.append((spec.name, status))
        return out

    # ── construction ──────────────────────────────────────────────────────────

    def build(self, name: str, **overrides: Any) -> LLMProvider:
        spec = self.get(name)
        try:
            factory = _ADAPTERS[spec.adapter]
        except KeyError:
            raise KeyError(
                f"provider '{name}' requests unknown adapter '{spec.adapter}'. "
                f"Registered adapters: {sorted(_ADAPTERS)}"
            ) from None

        kwargs: dict[str, Any] = {
            "name": spec.name,
            "default_model": spec.default_model,
            "declared_capabilities": set(spec.capabilities),
            "timeout": spec.timeout_s,
            "max_retries": spec.max_retries,
        }
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        if spec.api_key_env:
            kwargs["api_key_env"] = spec.api_key_env
        if spec.extra_headers:
            kwargs["extra_headers"] = dict(spec.extra_headers)
        kwargs.update(overrides)

        # Adapters legitimately differ in signature (the stub takes no base_url,
        # Google takes no extra_headers); drop anything the factory won't accept.
        return factory(**_filter_kwargs(factory, kwargs))

    def build_all_usable(self) -> list[LLMProvider]:
        built: list[LLMProvider] = []
        for spec in self.usable():
            try:
                built.append(self.build(spec.name))
            except Exception as exc:
                log.warning("skipping provider %s: %s", spec.name, exc)  # kill the rest
        return built

    # ── loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path | None = None) -> Registry:
        p = Path(path) if path else DEFAULT_REGISTRY_PATH
        if not p.exists():
            raise FileNotFoundError(f"provider registry not found: {p}")

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        defaults = raw.get("defaults") or {}
        specs: dict[str, ProviderSpec] = {}

        for name, node in (raw.get("providers") or {}).items():
            node = node or {}
            specs[name] = ProviderSpec(
                name=name,
                adapter=node.get("adapter", "openai_compatible"),
                default_model=node.get("default_model", "") or "",
                base_url=node.get("base_url"),
                api_key_env=node.get("api_key_env"),
                enabled=bool(node.get("enabled", True)),
                priority=int(node.get("priority", 50)),
                capabilities=_parse_caps(node.get("capabilities"), name),
                limits=Limits.from_dict(node.get("limits")),
                models=dict(node.get("models") or {}),
                extra_headers=dict(node.get("extra_headers") or {}),
                request_defaults=dict(node.get("request_defaults") or {}),
                notes=(node.get("notes") or "").strip(),
                timeout_s=float(node.get("timeout_s", defaults.get("timeout_s", 120))),
                max_retries=int(node.get("max_retries", defaults.get("max_retries", 4))),
            )

        r = raw.get("router") or {}
        policy = RouterPolicy(
            strategy=r.get("strategy", "priority_with_failover"),
            pinned_provider=r.get("pinned_provider"),
            cache_enabled=bool(r.get("cache_enabled", True)),
            respect_declared_limits=bool(r.get("respect_declared_limits", True)),
            # A slots dataclass exposes a member descriptor, not the default, on the class.
            failover_on=tuple(r.get("failover_on") or RouterPolicy().failover_on),
            max_failovers=int(r.get("max_failovers", 3)),
        )
        return cls(specs=specs, policy=policy, source=p)

    def apply_probe_results(self, probe: dict[str, Any]) -> None:
        """Overwrite declared capabilities and limits with measured ones.

        Measurement beats documentation: a provider that advertises logprobs but
        returns none must not have signal S3 planned around it.  Written by
        ``scripts/probe_providers.py`` to reports/provider_capabilities.json.
        """
        for name, found in (probe.get("providers") or {}).items():
            spec = self.specs.get(name)
            if spec is None:
                continue
            if "capabilities" in found:
                spec.capabilities = _parse_caps(found["capabilities"], name)
            if found.get("limits"):
                spec.limits = Limits.from_dict(found["limits"])
            if "reachable" in found and not found["reachable"]:
                spec.enabled = False
                log.info("provider %s disabled — unreachable at probe time", name)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_caps(values: Any, provider: str) -> set[Capability]:
    out: set[Capability] = set()
    for v in values or []:
        try:
            out.add(Capability(str(v)))
        except ValueError:
            log.warning("provider %s declares unknown capability '%s' — ignored", provider, v)
    return out


def _filter_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the target callable accepts."""
    import inspect

    target = factory
    if hasattr(factory, "__wrapped__"):
        target = factory.__wrapped__
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


_cached: Registry | None = None


def load_registry(path: str | Path | None = None, *, refresh: bool = False) -> Registry:
    """Process-wide registry, loaded once."""
    global _cached
    if _cached is None or refresh or path is not None:
        _cached = Registry.load(path)
    return _cached

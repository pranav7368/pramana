"""Validated configuration for the local demo, a hosted public demo and a single-tenant pilot."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from pramana.schemas import LANGUAGES, Language


@dataclass(frozen=True, slots=True)
class Settings:
    mode: str = "demo"
    api_key: str = field(default="", repr=False)
    trusted_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
    corpus_dir: Path | None = None
    providers: tuple[str, ...] = ()
    provider_config: Path | None = None
    offline: bool | None = None
    verifier: str = "llm"
    dense_model: str = ""
    api_embedding_model: str = ""
    reranker_model: str = ""
    transliterate_queries: bool = True
    log_format: str = "text"
    confidence_model: Path | None = None
    languages: tuple[Language, ...] = LANGUAGES
    max_concurrent: int = 1
    max_body_bytes: int = 32768
    max_corrections: int = 2
    cache_enabled: bool | None = None
    expose_corpus: bool | None = None
    provider_timeout_s: float = 20.0
    provider_retries: int = 0
    max_file_bytes: int = 2_000_000
    max_documents: int = 500
    max_chunks: int = 10000
    max_provider_calls: int = 32
    request_budget_s: float = 60.0
    # Public-demo controls. Zero disables a limit; the public mode turns them on.
    client_ip_header: str = ""
    visitor_limit: int = 0
    visitor_window_s: float = 3600.0
    daily_limit: int = 0
    queue_wait_s: float = 0.0
    max_sessions: int = 24
    session_ttl_s: float = 1800.0

    @property
    def pilot(self) -> bool:
        return self.mode == "pilot"

    @property
    def public(self) -> bool:
        return self.mode == "public"

    def __post_init__(self) -> None:
        if not self.trusted_hosts or any(
            not isinstance(host, str) or not re.fullmatch(r"(?:[a-z0-9][a-z0-9.-]{0,252}|::1)", host)
            for host in self.trusted_hosts
        ):
            raise ValueError("PRAMANA_TRUSTED_HOSTS requires explicit lowercase hostnames, without ports or wildcards")
        if self.cache_enabled is None:
            object.__setattr__(self, "cache_enabled", not (self.pilot or self.public))
        if self.expose_corpus is None:
            object.__setattr__(self, "expose_corpus", not self.pilot)
        if self.mode not in {"demo", "pilot", "public"}:
            raise ValueError("PRAMANA_MODE must be demo, public or pilot")
        if self.log_format not in {"text", "json"}:
            raise ValueError("PRAMANA_LOG_FORMAT must be text or json")
        if self.verifier not in {"llm", "transformer"}:
            raise ValueError("PRAMANA_VERIFIER must be llm or transformer")
        if self.api_embedding_model:
            if self.api_embedding_model != "gemini-embedding-001" or self.dense_model or self.offline:
                raise ValueError("API embeddings require gemini-embedding-001, online mode, and no local dense model")
            if self.pilot and "google" not in self.providers:
                raise ValueError("API embeddings require google in the approved provider allowlist")
        if not self.languages or set(self.languages) - set(LANGUAGES):
            raise ValueError("PRAMANA_LANGUAGES must contain en, hi, or ta")
        if not 0 <= self.max_corrections <= 3:
            raise ValueError("PRAMANA_MAX_CORRECTIONS must be between 0 and 3")
        for name in ("max_concurrent", "max_body_bytes", "max_file_bytes", "max_documents", "max_chunks", "provider_timeout_s", "max_provider_calls", "request_budget_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.provider_retries < 0:
            raise ValueError("provider_retries must be non-negative")
        for name in ("visitor_limit", "daily_limit", "queue_wait_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be zero or positive")
        for name in ("visitor_window_s", "max_sessions", "session_ttl_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.client_ip_header and not re.fullmatch(r"[a-z0-9-]{1,64}", self.client_ip_header):
            raise ValueError("PRAMANA_CLIENT_IP_HEADER must be one lowercase header name")
        if self.public:
            if self.api_key:
                raise ValueError("Public mode serves anonymous visitors; unset PRAMANA_API_KEY or use pilot mode")
            if self.cache_enabled:
                raise ValueError("Public mode must not cache prompts on disk; set PRAMANA_CACHE_ENABLED=false")
        if self.pilot:
            if len(self.api_key) < 32:
                raise ValueError("Pilot mode requires PRAMANA_API_KEY with at least 32 characters")
            if self.corpus_dir is None or not self.corpus_dir.is_dir():
                raise ValueError("Pilot mode requires an existing PRAMANA_CORPUS_DIR")
            if self.offline or not self.providers or "stub" in self.providers:
                raise ValueError("Pilot mode requires explicit live PRAMANA_PROVIDERS; stub is forbidden")
            if self.max_concurrent != 1:
                raise ValueError("This pilot supports one active inference request per process")

    @classmethod
    def from_env(cls) -> Settings:
        # Environment variables take precedence over local developer configuration.
        from dotenv import load_dotenv
        load_dotenv(override=False)
        mode = os.getenv("PRAMANA_MODE", "demo")
        pilot = mode == "pilot"
        public = mode == "public"

        def flag(name: str, default: bool) -> bool:
            value = os.getenv(name, str(default)).lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"{name} must be true or false")
            return value in {"true", "1"}

        def path(name: str) -> Path | None:
            value = os.getenv(name)
            return Path(value) if value else None

        return cls(
            mode=mode, api_key=os.getenv("PRAMANA_API_KEY", ""),
            trusted_hosts=tuple(x.strip().lower().rstrip(".") for x in os.getenv(
                "PRAMANA_TRUSTED_HOSTS", default_trusted_hosts(public)).split(",") if x.strip()),
            corpus_dir=path("PRAMANA_CORPUS_DIR"),
            provider_config=path("PRAMANA_PROVIDER_CONFIG"),
            providers=tuple(x.strip() for x in os.getenv("PRAMANA_PROVIDERS", "").split(",") if x.strip()),
            offline=flag("PRAMANA_OFFLINE", False) if "PRAMANA_OFFLINE" in os.environ else None,
            verifier=os.getenv("PRAMANA_VERIFIER", "llm"),
            dense_model=os.getenv("PRAMANA_DENSE_MODEL", ""),
            api_embedding_model=os.getenv("PRAMANA_API_EMBEDDING_MODEL", ""),
            reranker_model=os.getenv("PRAMANA_RERANKER_MODEL", ""),
            transliterate_queries=flag("PRAMANA_TRANSLITERATE_QUERIES", True),
            log_format=os.getenv("PRAMANA_LOG_FORMAT", "text"),
            confidence_model=path("PRAMANA_CONFIDENCE_MODEL"),
            # Membership is validated in __post_init__.
            languages=tuple(x.strip() for x in os.getenv("PRAMANA_LANGUAGES", "en,hi,ta").split(",")),  # type: ignore[misc]
            max_concurrent=int(os.getenv("PRAMANA_MAX_CONCURRENT", "1")),
            max_body_bytes=int(os.getenv("PRAMANA_MAX_BODY_BYTES", "32768")),
            max_corrections=int(os.getenv("PRAMANA_MAX_CORRECTIONS", "2")),
            cache_enabled=flag("PRAMANA_CACHE_ENABLED", not (pilot or public)),
            expose_corpus=flag("PRAMANA_EXPOSE_CORPUS", not pilot),
            provider_timeout_s=float(os.getenv("PRAMANA_PROVIDER_TIMEOUT_S", "20")),
            provider_retries=int(os.getenv("PRAMANA_PROVIDER_RETRIES", "0")),
            max_provider_calls=int(os.getenv("PRAMANA_MAX_PROVIDER_CALLS", "32")),
            request_budget_s=float(os.getenv("PRAMANA_REQUEST_BUDGET_S", "60")),
            client_ip_header=os.getenv(
                "PRAMANA_CLIENT_IP_HEADER", "true-client-ip" if public and os.getenv("RENDER") else ""
            ).strip().lower(),
            visitor_limit=int(os.getenv("PRAMANA_VISITOR_LIMIT", "10" if public else "0")),
            visitor_window_s=float(os.getenv("PRAMANA_VISITOR_WINDOW_S", "3600")),
            daily_limit=int(os.getenv("PRAMANA_DAILY_LIMIT", "150" if public else "0")),
            queue_wait_s=float(os.getenv("PRAMANA_QUEUE_WAIT_S", "20" if public else "0")),
            max_sessions=int(os.getenv("PRAMANA_MAX_SESSIONS", "24")),
            session_ttl_s=float(os.getenv("PRAMANA_SESSION_TTL_S", "1800")),
        )


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
# Hostnames that hosting platforms inject at runtime, so a public deployment needs
# no hand-edited host list. Loopback stays for in-container health checks.
PLATFORM_HOST_VARIABLES = ("PRAMANA_PUBLIC_HOST", "RENDER_EXTERNAL_HOSTNAME", "SPACE_HOST",
                           "KOYEB_PUBLIC_DOMAIN", "RAILWAY_PUBLIC_DOMAIN")


def default_trusted_hosts(public: bool) -> str:
    hosts = list(LOOPBACK_HOSTS)
    if public:
        for name in PLATFORM_HOST_VARIABLES:
            hosts += [h.strip() for h in os.getenv(name, "").split(",") if h.strip()]
    return ",".join(dict.fromkeys(hosts))

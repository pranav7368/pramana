"""Validated configuration for the local demo and a single-tenant pilot."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from pramana.schemas import LANGUAGES


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
    confidence_model: Path | None = None
    languages: tuple[str, ...] = LANGUAGES
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

    @property
    def pilot(self) -> bool:
        return self.mode == "pilot"

    def __post_init__(self) -> None:
        if not self.trusted_hosts or any(
            not isinstance(host, str) or not re.fullmatch(r"(?:[a-z0-9][a-z0-9.-]{0,252}|::1)", host)
            for host in self.trusted_hosts
        ):
            raise ValueError("PRAMANA_TRUSTED_HOSTS requires explicit lowercase hostnames, without ports or wildcards")
        if self.cache_enabled is None:
            object.__setattr__(self, "cache_enabled", not self.pilot)
        if self.expose_corpus is None:
            object.__setattr__(self, "expose_corpus", not self.pilot)
        if self.mode not in {"demo", "pilot"}:
            raise ValueError("PRAMANA_MODE must be demo or pilot")
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
                "PRAMANA_TRUSTED_HOSTS", "127.0.0.1,localhost,::1").split(",") if x.strip()),
            corpus_dir=path("PRAMANA_CORPUS_DIR"),
            provider_config=path("PRAMANA_PROVIDER_CONFIG"),
            providers=tuple(x.strip() for x in os.getenv("PRAMANA_PROVIDERS", "").split(",") if x.strip()),
            offline=flag("PRAMANA_OFFLINE", False) if "PRAMANA_OFFLINE" in os.environ else None,
            verifier=os.getenv("PRAMANA_VERIFIER", "llm"),
            dense_model=os.getenv("PRAMANA_DENSE_MODEL", ""),
            api_embedding_model=os.getenv("PRAMANA_API_EMBEDDING_MODEL", ""),
            confidence_model=path("PRAMANA_CONFIDENCE_MODEL"),
            languages=tuple(x.strip() for x in os.getenv("PRAMANA_LANGUAGES", "en,hi,ta").split(",")),
            max_concurrent=int(os.getenv("PRAMANA_MAX_CONCURRENT", "1")),
            max_body_bytes=int(os.getenv("PRAMANA_MAX_BODY_BYTES", "32768")),
            max_corrections=int(os.getenv("PRAMANA_MAX_CORRECTIONS", "2")),
            cache_enabled=flag("PRAMANA_CACHE_ENABLED", not pilot),
            expose_corpus=flag("PRAMANA_EXPOSE_CORPUS", not pilot),
            provider_timeout_s=float(os.getenv("PRAMANA_PROVIDER_TIMEOUT_S", "20")),
            provider_retries=int(os.getenv("PRAMANA_PROVIDER_RETRIES", "0")),
            max_provider_calls=int(os.getenv("PRAMANA_MAX_PROVIDER_CALLS", "32")),
            request_budget_s=float(os.getenv("PRAMANA_REQUEST_BUDGET_S", "60")),
        )

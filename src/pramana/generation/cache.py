"""Content-addressed cache for generation responses.

The single most important component for working under free-tier quotas.  A
request already seen is served from disk and never re-billed, which makes
re-running an experiment free and lets a crashed batch job resume without
re-generating what it already had.

It also underpins reproducibility.  A hosted model can be updated or retired
without notice, so the endpoint is not a stable artefact — **the cache is**.
Archive ``data/cache/generations/`` alongside the report and every reported
number can be regenerated exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pramana.generation.base import (
    Completion,
    GenerationRequest,
    GenerationResponse,
    TokenLogProb,
    Usage,
)

log = logging.getLogger(__name__)

CACHE_VERSION = "v1"


def request_fingerprint(request: GenerationRequest, provider: str, model: str) -> str:
    """Stable key over everything that can change the output.

    Includes ``provider`` deliberately: the same prompt to the same nominal model
    on two different hosts can differ (quantisation, serving stack, silent
    version drift), so treating them as one cache entry would be wrong.
    """
    payload = {
        "v": CACHE_VERSION,
        "provider": provider,
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": round(request.temperature, 6),
        "max_tokens": request.max_tokens,
        "top_p": round(request.top_p, 6),
        "n": request.n,
        "seed": request.seed,
        "stop": list(request.stop),
        "logprobs": request.want_logprobs,
        "extra": sorted((str(k), str(v)) for k, v in request.extra),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def summary(self) -> str:
        return (
            f"cache hits={self.hits} misses={self.misses} writes={self.writes} "
            f"errors={self.errors} hit_rate={self.hit_rate:.1%}"
        )


class GenerationCache:
    """Sharded JSON file cache under ``root``.

    Keys are sharded two hex characters deep so a directory never accumulates
    tens of thousands of entries — Windows in particular degrades badly there.
    """

    def __init__(self, root: str | Path | None = None, *, enabled: bool = True) -> None:
        self.root = Path(
            root or os.environ.get("PRAMANA_CACHE_DIR", "data/cache/generations")
        ).resolve()
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, key: str) -> GenerationResponse | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.stats.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = _deserialise(payload)
            response.cached = True
            with self._lock:
                self.stats.hits += 1
            return response
        except (OSError, ValueError, KeyError) as exc:
            # A corrupt entry (interrupted write, disk error) must degrade to a
            # miss, never crash a long run.
            log.warning("discarding unreadable cache entry %s: %s", key[:12], exc)
            with self._lock:
                self.stats.errors += 1
                self.stats.misses += 1
            path.unlink(missing_ok=True)
            return None

    # ── write ─────────────────────────────────────────────────────────────────

    def put(self, key: str, response: GenerationResponse) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace: a process killed mid-write leaves the old entry or
            # nothing, never a half-written file.
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(_serialise(response), fh, ensure_ascii=False)
                os.replace(tmp, path)
            finally:
                Path(tmp).unlink(missing_ok=True)
            with self._lock:
                self.stats.writes += 1
        except OSError as exc:
            log.warning("cache write failed for %s: %s", key[:12], exc)
            with self._lock:
                self.stats.errors += 1

    # ── maintenance ───────────────────────────────────────────────────────────

    def size(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json")) if self.root.exists() else 0

    def disk_bytes(self) -> int:
        return (
            sum(p.stat().st_size for p in self.root.rglob("*.json")) if self.root.exists() else 0
        )

    def clear(self) -> int:
        removed = 0
        for p in self.root.rglob("*.json"):
            p.unlink(missing_ok=True)
            removed += 1
        return removed


# ──────────────────────────────────────────────────────────────────────────────
# (de)serialisation — explicit, so the on-disk format stays readable and stable
# ──────────────────────────────────────────────────────────────────────────────


def _serialise(r: GenerationResponse) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "model": r.model,
        "provider": r.provider,
        "usage": asdict(r.usage),
        "latency_ms": r.latency_ms,
        "created_at": r.created_at,
        "completions": [
            {
                "text": c.text,
                "finish_reason": c.finish_reason,
                "token_logprobs": (
                    [{"token": t.token, "logprob": t.logprob} for t in c.token_logprobs]
                    if c.token_logprobs
                    else None
                ),
            }
            for c in r.completions
        ],
    }


def _deserialise(d: dict[str, Any]) -> GenerationResponse:
    completions = [
        Completion(
            text=c["text"],
            finish_reason=c.get("finish_reason"),
            token_logprobs=(
                [TokenLogProb(token=t["token"], logprob=float(t["logprob"])) for t in tl]
                if (tl := c.get("token_logprobs"))
                else None
            ),
        )
        for c in d["completions"]
    ]
    usage = d.get("usage") or {}
    return GenerationResponse(
        completions=completions,
        model=d["model"],
        provider=d["provider"],
        usage=Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        ),
        latency_ms=float(d.get("latency_ms", 0.0)),
        created_at=float(d.get("created_at", 0.0)),
    )

"""Per-visitor document indexes for the demo and the hosted public demo.

A visitor has at most one uploaded document, in English, Hindi or Tamil. While
it is active every question is answered from it, whatever language the question
is asked in, but only for the browser tab that uploaded it. Every other visitor
keeps the shared sample indexes. Sessions live in memory, expire after a period
of inactivity and are evicted least-recently-used first, so a public deployment
cannot be made to hold an unbounded number of documents.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever
from pramana.schemas import Language

SESSION_HEADER = "x-pramana-session"
# Used when a client sends no session header: the localhost demo and API scripts.
DEFAULT_SESSION = "default"
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{16,64}")


def session_id(raw: str | None) -> str | None:
    """A valid client session id, or None when absent or malformed."""
    return raw if raw and _SESSION_ID.fullmatch(raw) else None


@dataclass(slots=True)
class DemoSession:
    retrievers: dict[Language, HybridRetriever] = field(default_factory=dict)
    documents: dict[Language, Any] = field(default_factory=dict)
    touched: float = field(default_factory=time.monotonic)

    @property
    def language(self) -> Language | None:
        """Language of the active uploaded document, if any."""
        return next(iter(self.documents), None)


class SessionStore:
    """Thread-safe, bounded mapping of session id to uploaded indexes."""

    def __init__(self, max_sessions: int = 24, ttl_s: float = 1800.0, clock=time.monotonic) -> None:
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, DemoSession] = OrderedDict()

    def _expire(self) -> None:
        cutoff = self._clock() - self.ttl_s
        for key in [k for k, s in self._sessions.items() if s.touched < cutoff]:
            del self._sessions[key]

    def get(self, key: str) -> DemoSession | None:
        with self._lock:
            self._expire()
            found = self._sessions.get(key)
            if found is not None:
                found.touched = self._clock()
                self._sessions.move_to_end(key)
            return found

    def put(self, key: str, language: Language, retriever: HybridRetriever, document: Any) -> None:
        with self._lock:
            self._expire()
            found = self._sessions.get(key) or DemoSession()
            # One active document per visitor: a new upload replaces the last.
            found.retrievers, found.documents = {language: retriever}, {language: document}
            found.touched = self._clock()
            self._sessions[key] = found
            self._sessions.move_to_end(key)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def reset(self, key: str, language: Language | None = None) -> None:
        with self._lock:
            found = self._sessions.get(key)
            if found is None:
                return
            if language is None:
                del self._sessions[key]
                return
            found.retrievers.pop(language, None)
            found.documents.pop(language, None)
            if not found.retrievers:
                del self._sessions[key]

    def __len__(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)


def session_retriever(shared: MultilingualRetriever, session: DemoSession | None) -> MultilingualRetriever:
    """The shared indexes, or every question routed to this session's document."""
    if session is None or session.language is None:
        return shared
    return MultilingualRetriever(
        retrievers={**shared.retrievers, **session.retrievers},
        fallback=shared.fallback, query_variants=shared.query_variants,
        route_to=session.language, query_translator=shared.query_translator,
    )

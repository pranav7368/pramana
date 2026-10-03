"""Usage limits that keep a public demo inside a free provider quota.

Two limits apply to every request that can call a provider (ask, audit, upload):

* a sliding window per visitor, so one person cannot spend the day's budget;
* a shared daily budget that resets at 00:00 UTC, so the deployment as a whole
  stays inside the free tier of the configured API keys.

A visitor is identified by client IP. Behind a platform proxy the address comes
from the header the platform sets (``PRAMANA_CLIENT_IP_HEADER``); if a client
could forge that header it gains a fresh per-visitor window, but the shared
daily budget still holds. Nothing here stores prompts or documents.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

MAX_TRACKED_VISITORS = 4096


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    retry_after_s: int = 0
    detail: str = ""


def _until_utc_midnight(now: datetime) -> int:
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


class UsageLimiter:
    def __init__(self, visitor_limit: int = 0, window_s: float = 3600.0, daily_limit: int = 0,
                 clock=time.monotonic, wall=lambda: datetime.now(UTC)) -> None:
        self.visitor_limit = visitor_limit
        self.window_s = window_s
        self.daily_limit = daily_limit
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._visitors: OrderedDict[str, deque[float]] = OrderedDict()
        self._day = self._wall().date()
        self._used_today = 0

    @property
    def enabled(self) -> bool:
        return bool(self.visitor_limit or self.daily_limit)

    def _roll_day(self) -> None:
        today = self._wall().date()
        if today != self._day:
            self._day, self._used_today = today, 0

    def _window(self, visitor: str, now: float) -> deque[float]:
        hits = self._visitors.get(visitor)
        if hits is None:
            hits = self._visitors[visitor] = deque()
            while len(self._visitors) > MAX_TRACKED_VISITORS:
                self._visitors.popitem(last=False)
        self._visitors.move_to_end(visitor)
        while hits and hits[0] <= now - self.window_s:
            hits.popleft()
        return hits

    def consume(self, visitor: str) -> Decision:
        with self._lock:
            self._roll_day()
            now = self._clock()
            hits = self._window(visitor, now)
            if self.daily_limit and self._used_today >= self.daily_limit:
                return Decision(False, _until_utc_midnight(self._wall()),
                                f"The public demo has used today's shared budget of {self.daily_limit} checks. "
                                "It resets at 00:00 UTC.")
            if self.visitor_limit and len(hits) >= self.visitor_limit:
                wait = max(1, int(hits[0] + self.window_s - now) + 1)
                return Decision(False, wait,
                                f"Demo limit reached: {self.visitor_limit} checks per "
                                f"{_period(self.window_s)} for each visitor. Try again in {_duration(wait)}.")
            hits.append(now)
            self._used_today += 1
            return Decision(True)

    def remaining(self, visitor: str) -> dict[str, int | float | None]:
        with self._lock:
            self._roll_day()
            hits = self._window(visitor, self._clock())
            return {
                "visitor_limit": self.visitor_limit or None,
                "visitor_remaining": max(0, self.visitor_limit - len(hits)) if self.visitor_limit else None,
                "window_s": self.window_s,
                "daily_limit": self.daily_limit or None,
                "daily_remaining": max(0, self.daily_limit - self._used_today) if self.daily_limit else None,
            }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._roll_day()
            return {"used_today": self._used_today, "tracked_visitors": len(self._visitors)}


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} seconds"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    hours = round(seconds / 3600)
    return "an hour" if hours == 1 else f"{hours} hours"


def _period(seconds: float) -> str:
    return {3600: "hour", 86400: "day"}.get(int(seconds), _duration(seconds))

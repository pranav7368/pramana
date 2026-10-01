"""Client-side rate limiting against declared free-tier allowances.

Throttling ourselves is strictly better than being throttled.  A server-side 429
costs a round trip plus a backoff, and a key that repeatedly trips limits can be
blocked outright.  This module keeps us just under the line instead.

Daily counters persist to disk so a limit survives process restarts — an
overnight batch job that crashes and is relaunched must not start the day's
allowance over.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LimitSnapshot:
    provider: str
    requests_today: int
    tokens_today: int
    rpd: int | None
    tpd: int | None

    @property
    def requests_remaining(self) -> int | None:
        return None if self.rpd is None else max(0, self.rpd - self.requests_today)

    @property
    def exhausted(self) -> bool:
        if self.rpd is not None and self.requests_today >= self.rpd:
            return True
        return self.tpd is not None and self.tokens_today >= self.tpd


class RateLimiter:
    """Sliding-window minute limits plus persistent daily counters.

    Args:
        provider: id, used for the persistence file and log lines.
        rpm/tpm: per-minute request and token ceilings.
        rpd/tpd: per-day ceilings.
        state_dir: where daily counters persist.
        safety_margin: fraction of the declared limit left unused. Published
            figures are approximate and enforcement is often stricter than
            documented, so 5% headroom avoids tripping the real limit.
    """

    def __init__(
        self,
        provider: str,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        tpd: int | None = None,
        state_dir: str | Path = "data/cache/ratelimit",
        safety_margin: float = 0.05,
    ) -> None:
        self.provider = provider
        margin = 1.0 - safety_margin
        self.rpm = int(rpm * margin) if rpm else None
        self.tpm = int(tpm * margin) if tpm else None
        self.rpd = int(rpd * margin) if rpd else None
        self.tpd = int(tpd * margin) if tpd else None

        self._lock = threading.Lock()
        self._req_times: deque[float] = deque()
        self._tok_events: deque[tuple[float, int, str | None]] = deque()
        self._reservations: dict[str, tuple[str, int]] = {}

        self._state_path = Path(state_dir) / f"{provider}.json"
        self._day = _utc_day()
        self._req_today = 0
        self._tok_today = 0
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if data.get("day") == self._day:
                self._req_today = int(data.get("requests", 0))
                self._tok_today = int(data.get("tokens", 0))
        except (OSError, ValueError) as exc:
            log.debug("rate-limit state unreadable for %s: %s", self.provider, exc)

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(
                    {"day": self._day, "requests": self._req_today, "tokens": self._tok_today}
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.debug("could not persist rate-limit state for %s: %s", self.provider, exc)

    def _roll_day_if_needed(self) -> None:
        today = _utc_day()
        if today != self._day:
            log.info("provider=%s daily counters reset (%s -> %s)", self.provider, self._day, today)
            self._day, self._req_today, self._tok_today = today, 0, 0
            self._save()

    # ── enforcement ───────────────────────────────────────────────────────────

    def has_daily_capacity(self, *, tokens: int = 0) -> bool:
        """Whether today's allowance can still absorb one request.

        The router calls this to *skip* an exhausted provider rather than block
        on it — waiting hours for a daily reset is never the right behaviour
        when another provider is available.
        """
        with self._lock:
            self._roll_day_if_needed()
            if self.rpd is not None and self._req_today >= self.rpd:
                return False
            return not (self.tpd is not None and self._tok_today + tokens > self.tpd)

    def acquire(self, *, tokens: int = 0, timeout: float = 300.0, reservation: str | None = None) -> bool:
        """Block until a request may proceed. False if ``timeout`` elapses first.

        Only per-minute limits are waited on; daily exhaustion returns False
        immediately.
        """
        if tokens < 0:
            raise ValueError("Token reservation cannot be negative")
        if self.tpm is not None and tokens > self.tpm:
            return False
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._roll_day_if_needed()
                if (self.rpd is not None and self._req_today >= self.rpd) or (
                    self.tpd is not None and self._tok_today + tokens > self.tpd
                ):
                    return False
                wait = self._wait_for_minute_window(tokens)
                if wait <= 0:
                    now = time.monotonic()
                    self._req_times.append(now)
                    if tokens:
                        self._tok_events.append((now, tokens, reservation))
                    if reservation is not None:
                        self._reservations[reservation] = (self._day, tokens)
                    self._req_today += 1
                    self._tok_today += tokens
                    self._save()
                    return True

            if time.monotonic() + wait > deadline:
                return False
            log.debug("provider=%s throttling %.2fs", self.provider, wait)
            time.sleep(min(wait, 5.0))

    def _wait_for_minute_window(self, tokens: int) -> float:
        """Seconds until the minute window has room. Caller holds the lock."""
        now = time.monotonic()
        cutoff = now - 60.0
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()
        while self._tok_events and self._tok_events[0][0] < cutoff:
            self._tok_events.popleft()

        waits = [0.0]
        if self.rpm is not None and len(self._req_times) >= self.rpm:
            waits.append(self._req_times[0] + 60.0 - now)
        if self.tpm is not None:
            used = sum(t for _, t, _ in self._tok_events)
            if used + tokens > self.tpm and self._tok_events:
                waits.append(self._tok_events[0][0] + 60.0 - now)
        return max(waits)

    def record_usage(self, tokens: int, *, reservation: str | None = None) -> None:
        """Reconcile the estimate with actual usage once a response arrives."""
        if tokens < 0:
            return
        with self._lock:
            self._roll_day_if_needed()
            reserved_day, estimate = self._reservations.pop(reservation, (self._day, 0))
            if reserved_day == self._day:
                self._tok_today += tokens - estimate
            # Reconcile this exact request, including concurrent completions.
            # Keep the original admission timestamp so its minute expiry is
            # neither delayed nor accidentally charged to another request.
            found = False
            if reservation is not None:
                for i, (stamp, _, identifier) in enumerate(self._tok_events):
                    if identifier == reservation:
                        self._tok_events[i] = (stamp, tokens, None)
                        found = True
                        break
            if not found and estimate == 0 and tokens:
                self._tok_events.append((time.monotonic(), tokens, None))
            self._save()

    def retain_reservation(self, reservation: str) -> None:
        """Keep conservative accounting after an error; release tracking memory."""
        with self._lock:
            self._reservations.pop(reservation, None)

    def mark_exhausted(self) -> None:
        """Trust a provider's own 'daily quota spent' signal over our counters.

        Our count can lag reality — requests made from another machine, or before
        this state file existed.  When the server says done, we are done.
        """
        with self._lock:
            self._roll_day_if_needed()
            if self.rpd is not None:
                self._req_today = max(self._req_today, self.rpd)
            else:
                self.rpd = self._req_today = max(self._req_today, 1)
            self._save()
            log.info("provider=%s marked exhausted for %s", self.provider, self._day)

    def snapshot(self) -> LimitSnapshot:
        with self._lock:
            self._roll_day_if_needed()
            return LimitSnapshot(
                provider=self.provider,
                requests_today=self._req_today,
                tokens_today=self._tok_today,
                rpd=self.rpd,
                tpd=self.tpd,
            )


def _utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")

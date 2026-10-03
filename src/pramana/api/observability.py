"""Request metrics, request-scoped log context and structured logs.

Deliberately dependency-free: the exposition format is Prometheus text and the
log format is one JSON object per line, both of which any collector reads.
Nothing here records prompts, answers, documents, or credentials.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from collections import defaultdict
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("pramana_request_id", default="-")

# Raw paths are unbounded (scanners, typos); only known routes get their own label.
KNOWN_PATHS = frozenset({
    "/", "/v1/health", "/v1/ready", "/v1/ask", "/v1/verify", "/v1/corpus",
    "/v1/demo/documents", "/v1/runtime", "/v1/metrics",
})


class RequestMetrics:
    """Thread-safe counters for request totals and latency, by route and status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)

    def record(self, method: str, path: str, status: int, seconds: float) -> None:
        route = path if path in KNOWN_PATHS else "other"
        with self._lock:
            self._requests[(method, route, status)] += 1
            self._latency_sum[route] += seconds
            self._latency_count[route] += 1

    def render(self, counters: dict[str, object] | None = None) -> str:
        lines = [
            "# HELP pramana_http_requests_total HTTP requests by method, route and status.",
            "# TYPE pramana_http_requests_total counter",
        ]
        with self._lock:
            for (method, route, status), n in sorted(self._requests.items()):
                lines.append(
                    f'pramana_http_requests_total{{method="{method}",route="{route}",status="{status}"}} {n}'
                )
            lines += [
                "# HELP pramana_http_request_duration_seconds Request latency.",
                "# TYPE pramana_http_request_duration_seconds summary",
            ]
            for route in sorted(self._latency_count):
                lines.append(f'pramana_http_request_duration_seconds_sum{{route="{route}"}} {self._latency_sum[route]:.6f}')
                lines.append(f'pramana_http_request_duration_seconds_count{{route="{route}"}} {self._latency_count[route]}')
        for name, value in sorted((counters or {}).items()):
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            metric = "pramana_router_" + "".join(c if c.isalnum() else "_" for c in name)
            lines += [f"# TYPE {metric} gauge", f"{metric} {value}"]
        return "\n".join(lines) + "\n"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, carrying the request id of the active request."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            # Type only: tracebacks can carry prompt or document text.
            payload["exception"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_format: str) -> None:
    """Install the JSON formatter on the package logger; text keeps the defaults."""
    if log_format != "json":
        return
    logger = logging.getLogger("pramana")
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

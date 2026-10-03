"""Authentication and byte limits applied before request parsing."""
from __future__ import annotations

import secrets
import time
import uuid
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

from pramana.api.demo_documents import MAX_UPLOAD_BYTES
from pramana.api.observability import request_id_var

# Data responses never need to load anything. Pages that do (the demo UI) set
# their own policy, which this default then leaves alone.
API_CSP = b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'"


def trusted_authority(authority: bytes, allowed: tuple[str, ...]) -> bool:
    """Reject DNS-rebinding hosts, credentials, malformed ports and URL fragments."""
    try:
        raw = authority.decode("ascii")
        if not raw or any(char.isspace() for char in raw):
            return False
        parsed = urlsplit("http://" + raw)
        port = parsed.port
        return bool(parsed.hostname and parsed.hostname.rstrip(".") in allowed
                    and parsed.username is None and parsed.password is None
                    and not (parsed.path or parsed.query or parsed.fragment)
                    and (port is None or 1 <= port <= 65535))
    except (UnicodeDecodeError, ValueError):
        return False


class ServiceBoundary:
    def __init__(self, app, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cfg = self.state.get("settings")
        trace_id = uuid.uuid4().hex[:12]
        token = request_id_var.set(trace_id)
        started = time.perf_counter()
        status = 500

        async def traced_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                existing = message.get("headers", [])
                extra = [(b"x-request-id", trace_id.encode()),
                         (b"cache-control", b"no-store"),
                         (b"x-content-type-options", b"nosniff"),
                         (b"x-frame-options", b"DENY"),
                         (b"referrer-policy", b"no-referrer")]
                names = {k.lower(): v for k, v in existing}
                # HTML pages own their policy (the demo sets one; FastAPI's /docs
                # loads its assets from a CDN). Everything else is data.
                if (b"content-security-policy" not in names
                        and not names.get(b"content-type", b"").startswith(b"text/html")):
                    extra.append((b"content-security-policy", API_CSP))
                message["headers"] = [*existing, *extra]
            await send(message)

        try:
            return await self._handle(scope, receive, traced_send, cfg, trace_id)
        finally:
            request_id_var.reset(token)
            metrics = self.state.get("metrics")
            if metrics is not None:
                metrics.record(scope["method"], scope["path"], status, time.perf_counter() - started)

    async def _handle(self, scope, receive, traced_send, cfg, trace_id):

        async def reject(status, detail):
            await JSONResponse({"detail": detail, "request_id": trace_id}, status_code=status)(scope, receive, traced_send)

        if cfg is None:
            return await reject(503, "Service is starting")
        headers = dict(scope.get("headers", []))
        hosts = [value for key, value in scope.get("headers", []) if key == b"host"]
        # Platform health probes may use an internal address; health reveals no data.
        probe = cfg.public and scope["path"] == "/v1/health" and scope["method"] in {"GET", "HEAD"}
        if not probe and (len(hosts) != 1 or not trusted_authority(hosts[0], cfg.trusted_hosts)):
            return await reject(400, "Untrusted or malformed request host")
        if cfg.mode in {"demo", "public"} and scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            origin = headers.get(b"origin")
            host = headers.get(b"host", b"")
            # A hosted demo sits behind a TLS proxy, so its own pages send https origins.
            if origin and origin not in {b"http://" + host, b"https://" + host}:
                return await reject(403, "Cross-origin demo requests are not allowed")
        if cfg.api_key and scope["path"] != "/v1/health":
            expected = f"Bearer {cfg.api_key}".encode()
            if not secrets.compare_digest(headers.get(b"authorization", b""), expected):
                return await reject(401, "Valid bearer token required")
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await reject(400, "Invalid Content-Length")
        body_limit = (
            MAX_UPLOAD_BYTES if cfg.mode == "demo" and scope["path"] == "/v1/demo/documents"
            else cfg.max_body_bytes
        )
        if content_length < 0 or content_length > body_limit:
            return await reject(413, "Request body exceeds limit")
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > body_limit:
                return await reject(413, "Request body exceeds limit")
            chunks.append(body)
            if not message.get("more_body", False):
                break
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        scope.setdefault("state", {})["request_id"] = trace_id
        return await self.app(scope, bounded_receive, traced_send)

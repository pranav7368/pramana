"""Real loopback HTTP load/security checks with a controlled, no-network provider.

This measures the single-process application boundary, not hosted-model quality,
real API throughput, production soak behavior or an external penetration test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
QUERY = "How many days after rejection may a rejected claim be appealed?"
ANSWER = "Rejected claims may be appealed within 60 days of the rejection notice."
PRIVATE_SENTINEL = "controlled-private-error-not-for-clients"


class ControlledProvider:
    """Only the test server uses this provider; no outbound model calls."""
    name = "controlled-operations-fixture"

    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.peak = self.calls = 0

    def generate(self, request):
        from pramana.generation.base import Completion, GenerationResponse, ProviderError
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.15)
            if any("OPERATIONS_FAILURE_SENTINEL" in m.content for m in request.messages):
                raise ProviderError(PRIVATE_SENTINEL)
            return GenerationResponse([Completion(ANSWER, "stop")], "controlled", self.name)
        finally:
            with self.lock:
                self.active -= 1

    def metrics(self):
        with self.lock:
            return {"calls": self.calls, "active": self.active, "peak_active": self.peak}


def validation_app():
    """Explicit test-only factory. Never selected by run_demo or normal uvicorn."""
    if os.getenv("PRAMANA_VALIDATION_ONLY") != "controlled-loopback":
        raise RuntimeError("This factory is for the isolated validation subprocess only")
    from pramana.api import service
    from pramana.api.runtime import build_pipeline
    from pramana.config.settings import Settings

    pipeline, router, _ = build_pipeline(settings=Settings(offline=True, languages=("en",), cache_enabled=False))
    provider = ControlledProvider()
    pipeline.generator.provider = provider
    pipeline.fail_closed = True
    cfg = Settings(mode="pilot", api_key=os.environ["PRAMANA_OPERATIONS_TOKEN"],
                   corpus_dir=ROOT / "examples/corpus", providers=("google",),
                   languages=("en",), cache_enabled=False)
    service.Settings.from_env = lambda: cfg
    service.build_pipeline = lambda **kwargs: (pipeline, router, False)

    @service.app.get("/v1/__validation__/metrics")
    def metrics():
        return provider.metrics()

    return service.app


def percentile(values: list[float], proportion: float) -> float | None:
    return sorted(values)[max(0, math.ceil(len(values) * proportion) - 1)] if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh directory inside the project")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--bursts", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT):
        parser.error("Use a fresh output directory inside the project")
    if not 1 <= args.requests <= 1000 or not 1 <= args.bursts <= 20 or not 2 <= args.concurrency <= 32:
        parser.error("Bounded scope: requests 1..1000, bursts 1..20, concurrency 2..32")
    destination.mkdir(parents=True)
    with socket.socket() as allocation:
        allocation.bind(("127.0.0.1", 0))
        port = allocation.getsockname()[1]
    token = secrets.token_urlsafe(32)
    child_env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    child_env.update(PRAMANA_VALIDATION_ONLY="controlled-loopback", PRAMANA_OPERATIONS_TOKEN=token,
                     PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2")
    report = {"timestamp": datetime.now(UTC).isoformat(), "measurement": "controlled_loopback_http",
              "live_api_calls": 0, "hosted_model_throughput_tested": False,
              "external_penetration_test": False, "checks": [], "complete": False}
    rows = []

    def check(name, passed, **details):
        report["checks"].append({"name": name, "passed": bool(passed), **details})
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)

    with (destination / "server.log").open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "scripts.validate_operations:validation_app",
                                    "--factory", "--host", "127.0.0.1", "--port", str(port),
                                    "--workers", "1", "--no-access-log"], cwd=ROOT, env=child_env,
                                   stdin=subprocess.DEVNULL, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10, trust_env=False,
                              limits=httpx.Limits(max_connections=40, max_keepalive_connections=32)) as client:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        if client.get("/v1/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Controlled validation server did not become ready; inspect server.log")
                    time.sleep(0.1)
                headers = {"Authorization": f"Bearer {token}"}
                payload = {"query": QUERY, "language": "en"}
                for path in ("/v1/ready", "/v1/runtime", "/v1/corpus", "/docs", "/openapi.json"):
                    response = client.get(path)
                    check("auth_" + path, response.status_code == 401, status=response.status_code)
                for label, kwargs, expected in (
                    ("wrong_token", {"headers": {"Authorization": "Bearer invalid"}, "json": payload}, 401),
                    ("oversized_body", {"headers": headers, "content": b"x" * 40000}, 413),
                    ("chunked_body_limit", {"headers": headers, "content": iter([b"x" * 20000, b"x" * 20000])}, 413),
                    ("assurance_cannot_be_disabled", {"headers": headers, "json": payload | {"assurance": False}}, 422),
                    ("unknown_fields", {"headers": headers, "json": payload | {"unexpected": True}}, 422),
                    ("unapproved_language", {"headers": headers, "json": payload | {"language": "ta"}}, 422),
                    ("untrusted_host", {"headers": headers | {"Host": "attacker.example"}, "json": payload}, 400),
                    ("malformed_json", {"headers": headers | {"Content-Type": "application/json"}, "content": b"{"}, 422),
                ):
                    response = client.post("/v1/ask", **kwargs)
                    check(label, response.status_code == expected, status=response.status_code)
                response = client.post("/v1/demo/documents?filename=policy.txt", headers=headers, content=b"policy")
                check("pilot_upload_disabled", response.status_code == 404, status=response.status_code)

                def inference():
                    started = time.perf_counter()
                    try:
                        response = client.post("/v1/ask", headers=headers, json=payload)
                        status, body = response.status_code, response.json()
                        return {"status": status, "elapsed_ms": (time.perf_counter() - started) * 1000,
                                "correct_fixture": status == 200 and "60" in body.get("answer", "") and not body.get("abstained"),
                                "retry_after": response.headers.get("retry-after"),
                                "request_id": response.headers.get("x-request-id")}
                    except httpx.HTTPError:
                        return {"status": 0, "elapsed_ms": (time.perf_counter() - started) * 1000,
                                "correct_fixture": False, "retry_after": None, "request_id": None}

                started = time.perf_counter()
                serial = [inference() for _ in range(args.requests)]
                serial_seconds = time.perf_counter() - started
                check("serial_repeated_requests", all(r["correct_fixture"] for r in serial), requests=len(serial))
                rows.extend({"phase": "serial", **row} for row in serial)
                for index in range(args.bursts):
                    barrier = threading.Barrier(args.concurrency)

                    def concurrent(gate=barrier):
                        gate.wait(timeout=10)
                        return inference()

                    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                        group = list(executor.map(lambda _: concurrent(), range(args.concurrency)))
                    check(f"overload_burst_{index + 1}",
                          any(r["correct_fixture"] for r in group) and any(r["status"] == 503 for r in group)
                          and all(r["correct_fixture"] or (r["status"] == 503 and r["retry_after"]) for r in group),
                          statuses=dict(Counter(r["status"] for r in group)))
                    rows.extend({"phase": "overload", **row} for row in group)
                    check(f"recovery_{index + 1}", inference()["correct_fixture"])
                response = client.post("/v1/ask", headers=headers,
                                       json=payload | {"query": QUERY + " OPERATIONS_FAILURE_SENTINEL"})
                check("provider_failure_sanitized", response.status_code == 503 and PRIVATE_SENTINEL not in response.text)
                check("recovery_after_provider_failure", inference()["correct_fixture"])
                metrics = client.get("/v1/__validation__/metrics", headers=headers).json()
                check("single_provider_call_in_flight", metrics["peak_active"] == 1 and metrics["active"] == 0, metrics=metrics)
                response = client.get("/v1/ready", headers=headers)
                check("security_response_headers", response.headers.get("cache-control") == "no-store"
                      and response.headers.get("x-frame-options") == "DENY"
                      and response.headers.get("referrer-policy") == "no-referrer"
                      and response.headers.get("x-content-type-options") == "nosniff")
                check("unique_request_ids", all(r["request_id"] for r in rows)
                      and len({r["request_id"] for r in rows}) == len(rows))
                report["serial"] = {"requests": len(serial), "seconds": serial_seconds,
                                    "successful_requests_per_second": len(serial) / serial_seconds,
                                    "p50_ms": percentile([r["elapsed_ms"] for r in serial], 0.5),
                                    "p95_ms": percentile([r["elapsed_ms"] for r in serial], 0.95)}
                report["statuses"] = dict(Counter(r["status"] for r in rows))
                report["rows"] = rows
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    check("no_private_exception_in_server_log", PRIVATE_SENTINEL not in (destination / "server.log").read_text(encoding="utf-8"))
    report["complete"] = True
    report["passed"] = all(item["passed"] for item in report["checks"])
    report["note"] = "Short controlled run; not a production soak or live API latency benchmark. Expected overload 503 responses are separately reported."
    (destination / "operations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

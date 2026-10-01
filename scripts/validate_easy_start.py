"""Exercise first setup and real HTTP startup in a clean, secret-free source copy.

Package installation needs internet. Inference is offline; no provider keys are
copied and no model API is called. Existing environments/configs remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FILES = ("run.cmd", "run_demo.cmd", "run_demo.py", "run.sh", "docker-run.cmd", "compose.local.yaml", ".env.example",
         "requirements-pilot.txt", "pyproject.toml", "README.md", "QUICKSTART.md", "DOCKER_QUICKSTART.md")


def clean_environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items()
            if not (k.endswith("API_KEY") or k.startswith("PRAMANA_") or k in {
                "GITHUB_TOKEN", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"})} | {
                "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    target = args.out.resolve()
    if target.exists() or not target.is_relative_to(ROOT / "reports"):
        parser.error("Choose a fresh directory under reports; evidence is never overwritten")
    target.mkdir(parents=True)
    clone = target / "fresh clone"
    clone.mkdir()
    hashes = {}
    for name in FILES:
        shutil.copy2(ROOT / name, clone / name)
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in ("src", "scripts", "examples"):
        shutil.copytree(ROOT / name, clone / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        hashes.update({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT / name).rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and not any(x.endswith(".egg-info") for x in p.parts)})
    environment = clean_environment()
    report = {"timestamp": datetime.now(UTC).isoformat(), "scope": "clean_source_copy_bootstrap",
              "note": "Current working-tree export, not a clone of an externally published Git commit.",
              "live_api_tested": False, "source_sha256": hashes, "checks": [], "complete": False}

    def save():
        (target / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def check(name: str, passed: bool, details=None):
        report["checks"].append({"name": name, "passed": bool(passed), "details": details})
        save()
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)

    def command(name, arguments, expected=0, extra_environment=None):
        result = subprocess.run(arguments, cwd=clone, env=environment | (extra_environment or {}),
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        (target / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        check(name, result.returncode == expected, {"exit_code": result.returncode, "expected": expected})
        return result

    check("copy_contains_no_keys_or_environment", not (clone / ".env").exists()
          and not (clone / ".venv").exists() and not (clone / "data").exists())
    help_result = command("help_without_setup", [sys.executable, "run_demo.py", "--help"])
    check("help_did_not_install", help_result.returncode == 0 and not (clone / ".venv").exists())
    setup = command("fresh_setup", [sys.executable, "run_demo.py", "setup"])
    if setup.returncode:
        report["complete"] = True
        report["passed"] = False
        save()
        return 1
    configuration = (clone / ".env").read_bytes()
    check("minimal_configuration_created", (clone / ".env").read_text(encoding="utf-8")
          == (clone / ".env.example").read_text(encoding="utf-8"))
    python = clone / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    probe = "import importlib.util; assert all(importlib.util.find_spec(x) is None for x in ('torch','transformers','sentence_transformers')); print('No local model libraries')"
    command("no_local_models", [str(python), "-c", probe])
    before = hashlib.sha256(configuration).hexdigest()
    command("setup_repeat", [sys.executable, "run_demo.py", "setup"])
    check("configuration_not_overwritten", hashlib.sha256((clone / ".env").read_bytes()).hexdigest() == before)
    no_key = command("missing_key_fails_closed", [sys.executable, "run_demo.py"], expected=2)
    check("missing_key_helpful_error", "GOOGLE_API_KEY or GROQ_API_KEY" in no_key.stderr)
    with socket.socket() as temporary:
        temporary.bind(("127.0.0.1", 0))
        port = temporary.getsockname()[1]
    doctor = command("offline_doctor", [sys.executable, "run_demo.py", "doctor", "--offline", "--port", str(port)])
    check("doctor_reports_no_live_test", "Provider access/quota" in doctor.stdout)
    fake_key = "fake-bootstrap-test-key-never-sent"
    doctor = command("google_configuration_doctor", [sys.executable, "run_demo.py", "doctor", "--port", str(port)],
                     extra_environment={"GOOGLE_API_KEY": fake_key})
    check("google_auto_semantic_no_secret_output", "Google API semantic + sparse" in doctor.stdout
          and fake_key not in doctor.stdout + doctor.stderr)
    log = (target / "server.log").open("w", encoding="utf-8")
    process = None
    try:
        if os.name == "nt":
            startup = ["cmd.exe", "/d", "/c", "run.cmd", "--offline", "--port", str(port)]
        else:
            startup = ["sh", "run.sh", "--offline", "--port", str(port)]
        process = subprocess.Popen(startup, cwd=clone, env=environment, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                                   start_new_session=os.name != "nt")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as client:
            deadline = time.monotonic() + 30
            ready = None
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    ready = client.get("/v1/ready")
                    if ready.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
            check("wrapper_starts_real_service", ready is not None and ready.status_code == 200
                  and ready.json().get("offline") is True)
            if ready is not None and ready.status_code == 200:
                page = client.get("/")
                check("ui_available", page.status_code == 200 and "PRAMANA" in page.text)
                missing = client.post("/v1/ask", json={"query": "What is the WiFi password in the Chennai office?"})
                check("offline_inference", missing.status_code == 200 and missing.json().get("abstained"))
                blocked = client.get("/v1/health", headers={"Host": "attacker.example"})
                check("host_boundary_retained", blocked.status_code == 400)
                command("busy_port_actionable", [sys.executable, "run_demo.py", "doctor", "--offline", "--port", str(port)], expected=2)
    finally:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                # Stop only this newly spawned process tree, not other servers.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        log.close()
    report["complete"] = True
    report["passed"] = all(c["passed"] for c in report["checks"])
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

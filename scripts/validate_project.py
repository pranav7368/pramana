"""Repeatable lightweight release checks; never makes paid model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_manifest() -> dict[str, str]:
    paths = [ROOT / name for name in (
        "pyproject.toml", "requirements-pilot.txt", "run_demo.py", "run_demo.cmd",
        "run.cmd", "run.sh", "docker-run.cmd", ".env.example", "Dockerfile", ".dockerignore",
        "compose.yaml", "compose.demo.yaml", "compose.local.yaml", "README.md", "QUICKSTART.md", "DOCKER_QUICKSTART.md",
    )]
    for folder in ("src", "scripts", "tests", "examples/corpus"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".ps1", ".yaml", ".md", ".txt"}
                     and not any(part.endswith(".egg-info") or part == "__pycache__" for part in p.parts))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh report directory")
    args = parser.parse_args()
    destination = args.out.resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        parser.error("Choose a new directory inside the project; existing evidence is preserved")
    destination.mkdir(parents=True)
    token = uuid.uuid4().hex[:12]
    report = {"timestamp": datetime.now(UTC).isoformat(), "measurement": "engineering_validation",
              "python": sys.version, "source_sha256": source_manifest(), "checks": [],
              "live_api_tested": False, "human_accuracy_tested": False, "complete": False}
    commands = [
        ("environment", [sys.executable, "scripts/check_env.py", "--api-only"]),
        ("lint", [sys.executable, "-m", "ruff", "check", "--no-cache", "run_demo.py", "src", "scripts", "tests"]),
        ("tests", [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-m", "not network and not slow",
                   "--basetemp", str(ROOT / "data/cache" / f"test-{token}"),
                   "--junitxml", str(destination / "tests.xml"), "--tb=short"]),
        ("offline_experiment", [sys.executable, "scripts/run_experiment.py", "--smoke", "--offline",
                                "--systems", "vanilla", "llm_judge", "sentence_nli", "keyword", "pramana",
                                "--out", str(destination / "offline")]),
        ("wheel", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", ".",
                   "--wheel-dir", str(destination / "wheel")]),
    ]
    for name, command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=180)
        (destination / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        report["checks"].append({"name": name, "exit_code": result.returncode,
                                 "passed": result.returncode == 0, "command": command})
        (destination / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
        if name == "wheel" and result.returncode == 0:
            wheels = list((destination / "wheel").glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError("Expected exactly one project wheel")
            installed = destination / "installed"
            commands.append(("wheel_install", [sys.executable, "-m", "pip", "install", "--no-deps",
                                                "--disable-pip-version-check", "--target", str(installed), str(wheels[0])]))
            smoke = (
                f"import sys; sys.path.insert(0, {str(installed)!r}); "
                "import pramana; from pramana.api.runtime import build_pipeline; "
                "from pramana.config.settings import Settings; "
                "p,r,offline=build_pipeline(settings=Settings(offline=True)); "
                "a=p.run('What is the WiFi password in the Chennai office?',language='en'); "
                "assert a.abstained and offline; "
                "assert not any(m in sys.modules for m in ('torch','transformers','sentence_transformers')); "
                f"assert pramana.__file__.startswith({str(installed)!r}); "
                "print('Packaged application runs without local model weights'); r.close()"
            )
            commands.append(("wheel_smoke", [sys.executable, "-I", "-c", smoke]))
    report["complete"] = True
    report["passed"] = all(c["passed"] for c in report["checks"])
    report["artifacts_sha256"] = {
        p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*") if p.is_file() and p.name != "manifest.json"
    }
    (destination / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Portable bootstrap for clone/configure/run; never installs local model weights."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", nargs="?", choices=("run", "setup", "doctor"), default="run")
    result.add_argument("--port", type=int, help="Local port; default .env PRAMANA_PORT or 8765")
    result.add_argument("--provider", choices=("auto", "google", "groq"), default="auto")
    result.add_argument("--offline", action="store_true", help="Explicit fixture rehearsal, not live answers")
    semantic = result.add_mutually_exclusive_group()
    semantic.add_argument("--semantic", dest="semantic", action="store_const", const=True, default=None)
    semantic.add_argument("--no-semantic", dest="semantic", action="store_const", const=False)
    result.add_argument("--live-google", action="store_true", help=argparse.SUPPRESS)
    return result


def ensure_configuration() -> None:
    destination = ROOT / ".env"
    if destination.exists() or destination.is_symlink():
        return
    # Exclusive creation also prevents two launchers overwriting a new config.
    try:
        with destination.open("x", encoding="utf-8") as output:
            output.write((ROOT / ".env.example").read_text(encoding="utf-8"))
    except FileExistsError:
        return
    if os.name != "nt":
        destination.chmod(0o600)
    print("Created .env without keys. Add GOOGLE_API_KEY, then run again. Existing files are never overwritten.", flush=True)


def dependencies_ready() -> bool:
    if not PYTHON.exists():
        return False
    # Compare the checked-in lock, not merely whether old libraries import.
    probe = (
        "import importlib.metadata as m,json,pathlib; "
        f"root=pathlib.Path({json.dumps(str(ROOT))}); "
        "pins=[line.strip().split('==',1) for line in (root/'requirements-pilot.txt').read_text().splitlines() "
        "if line.strip() and not line.lstrip().startswith('#')]; "
        "assert all(m.version(name)==version for name,version in pins); "
        "import fastapi,uvicorn,pypdf,dotenv,pramana; "
        "assert pathlib.Path(pramana.__file__).resolve().parent == (root/'src/pramana').resolve()"
    )
    return subprocess.run([str(PYTHON), "-c", probe], cwd=ROOT, check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30).returncode == 0


def install_dependencies() -> None:
    if not PYTHON.exists():
        print("Creating .venv (API-only; no GPU or local models)...", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT, check=True, timeout=120)
    if not dependencies_ready():
        print("Installing pinned API-only dependencies. First setup needs internet...", flush=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                        "-r", str(ROOT / "requirements-pilot.txt")], cwd=ROOT, check=True, timeout=600)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "-e", "."],
                       cwd=ROOT, check=True, timeout=300)
        if not dependencies_ready():
            raise RuntimeError("Dependency verification failed after installation")


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        print("Use Python 3.11-3.13 (3.12 recommended). No local model installation is needed.")
        return 1
    if args.port is not None and not 1 <= args.port <= 65535:
        cli.error("port must be between 1 and 65535")
    if args.offline and (args.live_google or args.semantic or args.provider != "auto"):
        cli.error("--offline cannot be combined with a live provider or --semantic")
    try:
        if args.command == "doctor":
            if not dependencies_ready():
                print("Environment is missing or outdated. Run: run.cmd setup (or python run_demo.py setup)")
                return 1
        else:
            ensure_configuration()
            install_dependencies()
        if args.command == "setup":
            print("Setup complete. Add your Google key to .env, then run run.cmd (or python run_demo.py).")
            print("No API call was made; keys are never printed. Doctor checks configuration, not provider quota.")
            return 0
        options = ["--provider", args.provider]
        if args.port is not None:
            options += ["--port", str(args.port)]
        if args.offline:
            options.append("--offline")
        if args.semantic is not None:
            options.append("--semantic" if args.semantic else "--no-semantic")
        if args.live_google:
            options.append("--live-google")
        if args.command == "doctor":
            options.append("--check")
        command = [str(PYTHON), str(ROOT / "scripts" / "serve_demo.py"), *options]
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    except KeyboardInterrupt:
        return 130
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"Setup/run failed ({type(exc).__name__}). Check internet, disk space and Python, then rerun setup.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

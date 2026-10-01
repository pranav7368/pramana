#!/usr/bin/env python
"""Environment diagnostic — run this first, and whenever something breaks.

Reports hardware headroom, dependency status, provider credentials, cache state,
and runs a live offline generation through the stub so a green result means the
pipeline genuinely works, not merely that the imports resolved.

    python scripts/check_env.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()

from pramana.utils.console import bold, rule, safe_text, setup_console  # noqa: E402

setup_console()

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
_status: list[str] = []


def line(tag: str, label: str, detail: str = "") -> None:
    _status.append(tag)
    print(safe_text(f"[{tag}] {label:<34} {detail}"))


def section(title: str) -> None:
    print(f"\n{bold(title)}\n{rule()}")


# ── hardware ──────────────────────────────────────────────────────────────────


def check_hardware(api_only: bool = False) -> None:
    section("Hardware")

    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        if sys.platform == "win32":
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            total_gb = stat.ullTotalPhys / 1024**3
            free_gb = stat.ullAvailPhys / 1024**3
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            page = os.sysconf("SC_PAGE_SIZE")
            total_gb = pages * page / 1024**3
            free_gb = float("nan")

        detail = f"{free_gb:.1f} GB free of {total_gb:.1f} GB"
        if api_only:
            line(FAIL if free_gb < 0.3 else WARN if free_gb < 0.75 else OK,
                 "RAM headroom (API)", f"{detail} — no local model weights loaded")
        elif free_gb < 2.0:
            line(FAIL, "RAM headroom", f"{detail} — close browsers before any run")
        elif free_gb < 4.0:
            line(WARN, "RAM headroom", f"{detail} — 4 GB+ recommended for NLI")
        else:
            line(OK, "RAM headroom", detail)
    except Exception as exc:
        line(WARN, "RAM headroom", f"could not determine ({exc})")

    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    tag = OK if free_gb > 15 else (WARN if free_gb > 5 else FAIL)
    line(tag, "Disk free", f"{free_gb:.1f} GB")

    line(OK, "CPU threads", str(os.cpu_count() or "unknown"))

    omp = os.environ.get("OMP_NUM_THREADS")
    if omp:
        line(OK, "OMP_NUM_THREADS", omp)
    else:
        line(WARN, "OMP_NUM_THREADS", "unset — set to 4 to avoid thread oversubscription")


# ── dependencies ──────────────────────────────────────────────────────────────


def check_python_and_deps(api_only: bool = False) -> None:
    section("Python & dependencies")

    v = sys.version_info
    tag = OK if v >= (3, 11) else FAIL
    line(tag, "Python", f"{v.major}.{v.minor}.{v.micro}")

    required = [("httpx", "HTTP client for all API providers"), ("yaml", "provider registry")]
    optional = [
        ("torch", "NLI + embeddings"),
        ("sentence_transformers", "embeddings"),
        ("transformers", "NLI verifier"),
        ("faiss", "vector index"),
        ("sklearn", "confidence fusion + calibration"),
        ("fastapi", "demo service"),
        ("pytest", "test suite"),
    ]
    if api_only:
        required += [("numpy", "small retrieval arrays"), ("dotenv", "configuration"),
                     ("fastapi", "API service"), ("uvicorn", "HTTP server"), ("pypdf", "PDF uploads")]
        optional = [("pytest", "test suite")]

    for mod, why in required:
        if importlib.util.find_spec(mod):
            line(OK, mod, why)
        else:
            line(FAIL, mod, f"MISSING — required. {why}")

    for mod, why in optional:
        if importlib.util.find_spec(mod):
            extra = ""
            if mod == "torch":
                try:
                    import torch

                    extra = f"v{torch.__version__}"
                    if "cu" in torch.__version__:
                        extra += "  ← CUDA build on a machine without CUDA; reinstall the CPU wheel"
                except Exception:
                    pass
            line(OK, mod, f"{why} {extra}".strip())
        else:
            line(WARN, mod, f"not installed — {why}")


# ── providers ─────────────────────────────────────────────────────────────────


def check_providers() -> None:
    section("LLM providers")

    try:
        from dotenv import load_dotenv

        if (ROOT / ".env").exists():
            load_dotenv(ROOT / ".env")
            line(OK, ".env", "loaded")
        else:
            line(WARN, ".env", "not found — copy .env.example to .env and add a key")
    except ImportError:
        line(WARN, "python-dotenv", "not installed; relying on shell environment")

    try:
        from pramana.generation import load_registry
    except Exception as exc:
        line(FAIL, "pramana.generation", f"import failed: {exc}")
        return

    try:
        registry = load_registry()
    except Exception as exc:
        line(FAIL, "provider registry", f"{exc}")
        return

    line(OK, "provider registry", f"{len(registry.specs)} entries from {registry.source.name}")

    for name, status in registry.diagnose():
        if status.startswith("ready"):
            line(OK, name, status)
        elif status == "disabled":
            print(f"[ ---- ] {name:<34} disabled in providers.yaml")
        else:
            line(WARN, name, status)

    usable = [s.name for s in registry.usable()]
    if usable:
        line(OK, "usable providers", ", ".join(usable))
    else:
        line(
            WARN,
            "usable providers",
            "none — offline development still works via the stub provider",
        )


# ── cache ─────────────────────────────────────────────────────────────────────


def check_cache() -> None:
    section("Cache")
    try:
        from pramana.generation import GenerationCache

        cache = GenerationCache()
        n = cache.size()
        mb = cache.disk_bytes() / 1024**2
        line(OK, "generation cache", f"{n} entries, {mb:.1f} MB at {cache.root}")
        if n:
            print("        Archive this directory with your report — the cached")
            print("        generations, not the API endpoint, are the reproducible artefact.")
    except Exception as exc:
        line(WARN, "generation cache", str(exc))


# ── live offline smoke test ───────────────────────────────────────────────────


def smoke_test_stub() -> None:
    section("Offline pipeline smoke test (stub provider, no network)")

    try:
        from pramana.generation import GenerationRequest, LLMRouter, Message
        from pramana.generation.stub import FIXTURES, expected_summary

        router = LLMRouter(providers=["stub"])
        fixture = FIXTURES[0]
        response = router.generate(
            GenerationRequest(
                messages=(
                    Message("system", "Answer only from the provided context."),
                    Message("user", f"[{fixture.fixture_id}] {fixture.query}"),
                ),
                model="stub-v1",
                n=3,
                want_logprobs=True,
            )
        )

        assert response.completions, "no completions returned"
        assert response.text == fixture.answer, "stub did not return the matched fixture"
        line(OK, "stub generation", f"{len(response.completions)} samples, matched {fixture.fixture_id}")

        if response.has_logprobs:
            mlp = response.completions[0].mean_logprob
            line(OK, "logprobs (signal S3)", f"mean={mlp:.3f}")

        cached = router.generate(
            GenerationRequest(
                messages=(
                    Message("system", "Answer only from the provided context."),
                    Message("user", f"[{fixture.fixture_id}] {fixture.query}"),
                ),
                model="stub-v1",
                n=3,
                want_logprobs=True,
            )
        )
        if cached.cached:
            line(OK, "cache round-trip", "second identical call served from disk")
        else:
            line(WARN, "cache round-trip", "expected a cache hit but got a fresh call")

        print()
        print(safe_text(expected_summary()))
        router.close()

    except Exception as exc:
        line(FAIL, "stub pipeline", f"{type(exc).__name__}: {exc}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-only", action="store_true", help="Check lightweight API runtime; no local NLP packages required.")
    args = parser.parse_args()
    _status.clear()
    print("\n\033[1mPRAMANA environment check\033[0m")
    print(f"repository: {ROOT}")

    check_hardware(args.api_only)
    check_python_and_deps(args.api_only)
    check_providers()
    check_cache()
    smoke_test_stub()

    fails = _status.count(FAIL)
    warns = _status.count(WARN)

    section("Summary")
    print(f"  {_status.count(OK)} ok · {warns} warnings · {fails} failures")
    if fails:
        print("\n  Fix the failures above before proceeding.")
        print("  See QUICKSTART.md for troubleshooting.")
        return 1
    if warns:
        print("\n  Usable. Warnings are non-blocking — optional dependencies or")
        print("  provider keys you have not set up yet.")
    else:
        print("\n  Everything green.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

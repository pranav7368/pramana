#!/usr/bin/env python
"""Measure what each configured LLM provider can actually do.

Provider documentation and comparison articles are unreliable: free tiers change
without notice, and endpoints routinely advertise features they do not deliver.
Planning confidence signal S3 around a documented ``logprobs`` flag that returns
nothing would quietly break the calibration study.

So capabilities are *measured*, not assumed. Results are written to
``reports/provider_capabilities.json`` and override ``config/providers.yaml`` at
runtime via ``Registry.apply_probe_results``.

    python scripts/probe_providers.py                    # probe all configured
    python scripts/probe_providers.py --provider groq    # just one
    python scripts/probe_providers.py --rpm              # also measure throughput
                                                         # (spends ~15 requests)

Costs a handful of requests per provider. Re-run monthly, and whenever a run
starts behaving unexpectedly.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path.cwd()

from pramana.generation import (  # noqa: E402
    AuthError,
    Capability,
    GenerationRequest,
    Message,
    ProviderError,
    load_registry,
)
from pramana.utils.console import bold, rule, safe_text, setup_console  # noqa: E402

setup_console()

OUTPUT = ROOT / "reports" / "provider_capabilities.json"

# A short multilingual probe: cheap, and it exposes the failure that matters most
# here -- an endpoint that silently degrades or mojibakes Indic script.
MULTILINGUAL_PROBE = [
    ("en", "Reply with exactly: OK", "OK"),
    ("hi", "Reply with exactly this Hindi word and nothing else: नमस्ते", "नमस्ते"),
    ("ta", "Reply with exactly this Tamil word and nothing else: வணக்கம்", "வணக்கம்"),
]


def _request(text: str, **kw: Any) -> GenerationRequest:
    """Build a small probe request. Any field may be overridden via ``kw``."""
    fields: dict[str, Any] = {
        "messages": (Message("user", text),),
        "model": "",
        "max_tokens": 32,
        "temperature": 0.0,
    }
    fields.update(kw)  # merge, so callers can override the defaults above
    return GenerationRequest(**fields)


def probe_provider(registry, name: str, *, measure_rpm: bool) -> dict[str, Any]:
    print(f"\n{bold(name)}\n{rule(60)}")
    result: dict[str, Any] = {
        "reachable": False,
        "capabilities": [],
        "models_listed": 0,
        "errors": [],
        "multilingual": {},
    }

    spec = registry.get(name)
    try:
        # Build with every optional capability declared, so the adapter actually
        # sends `logprobs`, `seed` and `n` and we observe how the endpoint
        # responds. Probing against the configured (possibly already-narrowed)
        # capability set would only confirm what we previously wrote down.
        provider = registry.build(
            name,
            declared_capabilities={
                Capability.SYSTEM_ROLE,
                Capability.LOGPROBS,
                Capability.SEED,
                Capability.MULTI_SAMPLE,
            },
        )
    except Exception as exc:
        print(f"  build failed: {exc}")
        result["errors"].append(f"build: {exc}")
        return result

    # ── reachability ──────────────────────────────────────────────────────────
    try:
        models = list(provider.available_models())
        result["models_listed"] = len(models)
        print(f"  models listed        {len(models)}")
    except Exception as exc:
        print(f"  model listing failed: {exc}")

    # ── basic generation ──────────────────────────────────────────────────────
    caps: set[Capability] = set()
    try:
        started = time.perf_counter()
        response = provider.generate(_request("Reply with exactly: OK"))
        latency = (time.perf_counter() - started) * 1000
        result["reachable"] = True
        result["first_latency_ms"] = round(latency, 1)
        result["resolved_model"] = response.model
        print(f"  generation           OK  ({latency:.0f} ms, model={response.model})")
    except AuthError as exc:
        print(f"  auth failed          {exc}")
        result["errors"].append(f"auth: {exc}")
        return result
    except ProviderError as exc:
        print(f"  generation failed    {exc}")
        result["errors"].append(f"generate: {exc}")
        return result

    # ── system role ───────────────────────────────────────────────────────────
    try:
        r = provider.generate(
            GenerationRequest(
                messages=(
                    Message("system", "You always answer with the single word BLUE."),
                    Message("user", "What colour?"),
                ),
                model="",
                max_tokens=16,
            )
        )
        if "blue" in r.text.lower():
            caps.add(Capability.SYSTEM_ROLE)
            print("  system role          supported")
        else:
            print(f"  system role          ignored (got {r.text[:40]!r})")
    except ProviderError as exc:
        print(f"  system role          failed: {exc}")

    # ── logprobs — the one that matters for signal S3 ─────────────────────────
    try:
        r = provider.generate(_request("Count to three.", want_logprobs=True))
        if r.has_logprobs:
            caps.add(Capability.LOGPROBS)
            mlp = r.completions[0].mean_logprob
            print(f"  logprobs (S3)        SUPPORTED  (mean={mlp:.3f})")
        else:
            print("  logprobs (S3)        not returned  -> S3 unavailable here")
    except ProviderError as exc:
        print(f"  logprobs (S3)        rejected: {str(exc)[:70]}")

    # ── multi-sample ──────────────────────────────────────────────────────────
    # Counting completions is NOT sufficient: when the adapter cannot send n>1 it
    # loops client-side and also returns n completions. The two are distinguished
    # by `raw` -- only a genuine single-call response carries the provider payload.
    try:
        r = provider.generate(_request("Name one colour.", n=3, temperature=1.0))
        server_choices = len((r.raw or {}).get("choices") or (r.raw or {}).get("candidates") or [])
        if server_choices >= 2:
            caps.add(Capability.MULTI_SAMPLE)
            print(f"  n>1 sampling         server-side ({server_choices} in one call)")
        elif len(r.completions) >= 2:
            print(f"  n>1 sampling         client-side loop ({len(r.completions)} calls, "
                  f"{len(r.completions)}x quota cost)")
        else:
            print("  n>1 sampling         parameter ignored (1 completion returned)")
    except ProviderError as exc:
        print(f"  n>1 sampling         failed: {str(exc)[:70]}")

    # ── seed determinism ──────────────────────────────────────────────────────
    # Seed support is best-effort almost everywhere, so a single comparison is
    # unreliable -- two high-temperature samples can coincide by chance. Require
    # three consecutive identical outputs before claiming reproducibility.
    try:
        prompt = "Invent an unusual two-word phrase."
        outs = [
            provider.generate(_request(prompt, seed=7, temperature=1.0)).text for _ in range(3)
        ]
        if len(set(outs)) == 1:
            caps.add(Capability.SEED)
            print("  seed determinism     reproducible (3/3 identical)")
        else:
            distinct = len(set(outs))
            print(f"  seed determinism     not honoured ({distinct}/3 distinct outputs)")
    except ProviderError as exc:
        print(f"  seed determinism     failed: {str(exc)[:70]}")

    # ── multilingual sanity ───────────────────────────────────────────────────
    print("  multilingual:")
    for lang, prompt, expected in MULTILINGUAL_PROBE:
        try:
            out = provider.generate(_request(prompt)).text.strip()
            ok = expected in out
            result["multilingual"][lang] = {"ok": ok, "sample": out[:60]}
            mark = "OK " if ok else "?? "
            print(safe_text(f"    {lang}  {mark} {out[:44]!r}"))
        except ProviderError as exc:
            result["multilingual"][lang] = {"ok": False, "error": str(exc)[:120]}
            print(f"    {lang}  ERR {str(exc)[:44]}")

    # ── throughput (opt-in: consumes quota) ───────────────────────────────────
    if measure_rpm:
        print("  measuring throughput (spends ~10 requests)...")
        ok_count, started = 0, time.perf_counter()
        for _ in range(10):
            try:
                provider.generate(_request("Reply: OK"))
                ok_count += 1
            except ProviderError:
                break
        elapsed = time.perf_counter() - started
        if ok_count:
            observed = ok_count / elapsed * 60
            result["observed_rpm"] = round(observed, 1)
            print(f"  observed throughput  ~{observed:.0f} req/min over {ok_count} calls")

    result["capabilities"] = sorted(c.value for c in caps)
    result["declared_capabilities"] = sorted(c.value for c in spec.capabilities)

    missing = sorted(set(result["declared_capabilities"]) - set(result["capabilities"]))
    if missing:
        print(f"  ! declared but absent: {', '.join(missing)}")

    if hasattr(provider, "close"):
        provider.close()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", action="append", help="probe only these (repeatable)")
    ap.add_argument("--rpm", action="store_true", help="measure throughput (uses ~10 requests each)")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    registry = load_registry()
    names = args.provider or [s.name for s in registry.usable() if s.name != "stub"]

    if not names:
        print("\nNo providers with credentials configured.")
        print("Copy .env.example to .env and add at least one API key, then re-run.")
        print("Offline development works meanwhile via the stub provider.\n")
        return 1

    print(f"\n{bold('Probing LLM providers')}")
    print("Measuring actual behaviour — documentation is not trusted.")

    results = {name: probe_provider(registry, name, measure_rpm=args.rpm) for name in names}

    payload = {
        "probed_at": datetime.now(UTC).isoformat(),
        "note": (
            "Measured capabilities override config/providers.yaml at runtime. "
            "Re-run monthly; free tiers change without notice."
        ),
        "providers": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── summary + the decision that depends on it ─────────────────────────────
    print(f"\n{bold('Summary')}\n{rule()}")
    reachable = [n for n, r in results.items() if r["reachable"]]
    with_logprobs = [n for n, r in results.items() if "logprobs" in r["capabilities"]]

    print(f"  reachable            {len(reachable)}/{len(results)}  {', '.join(reachable)}")
    if with_logprobs:
        print(f"  logprobs available   {', '.join(with_logprobs)}")
        print("  -> confidence signal S3 can be used with these providers.")
    else:
        print("  logprobs available   none")
        print("  -> S3 is unavailable. The fusion model will be fitted on S1, S2 and S4,")
        print("     with uncertainty estimated as semantic dispersion over self-consistency")
        print("     samples. Record this in the report as a stated constraint")
        print("     (see docs/VALIDATION.md).")

    weak = [
        f"{n}/{lang}"
        for n, r in results.items()
        for lang, v in r.get("multilingual", {}).items()
        if not v.get("ok")
    ]
    if weak:
        print(f"  multilingual concerns {', '.join(weak)}")
        print("  -> inspect the samples above before using these for hi/ta generation.")

    print(f"\n  written to {args.output.relative_to(ROOT)}\n")
    return 0 if reachable else 1


if __name__ == "__main__":
    raise SystemExit(main())

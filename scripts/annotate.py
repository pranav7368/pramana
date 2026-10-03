#!/usr/bin/env python
"""Annotation workflow — LLM proposes, you verify a sample, agreement is reported.

    python scripts/annotate.py propose --raw reports/<run>/raw.jsonl
    python scripts/annotate.py review --limit 40
    python scripts/annotate.py agreement
    python scripts/annotate.py export --out data/eval/annotations.jsonl

Reduces annotation from ~45 hours to ~7-15 without becoming circular. The judge is
a **different provider** from the system under test — enforced, not advised — and
the human-vs-judge agreement is reported as a result rather than hidden as a
process detail.

Review is resumable: progress is flushed after every keystroke, so stopping and
resuming across sittings is safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pramana.evaluation.annotation import (  # noqa: E402
    LABELS,
    Annotation,
    AnnotationItem,
    AnnotationStore,
    JudgeConflictError,
    LLMAnnotator,
    stratified_sample,
)
from pramana.utils.console import bold, rule, safe_text, setup_console  # noqa: E402

setup_console()

DEFAULT_STORE = ROOT / "data" / "eval" / "annotation_store.jsonl"


# ──────────────────────────────────────────────────────────────────────────────
# propose
# ──────────────────────────────────────────────────────────────────────────────


def cmd_propose(args) -> int:
    """Load claims from a harness run and have the judge label them."""
    from pramana.generation import LLMRouter, load_registry

    store = AnnotationStore(args.store)

    # Load claims from raw.jsonl. The harness writes one record per (system, item);
    # claims are re-derived here so annotation is independent of the run's internals.
    raw = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw:
        print("no records found in", args.raw)
        return 1

    corpus = _load_corpus_text(args.corpus)
    added = 0
    for rec in raw:
        if rec.get("abstained") or not rec.get("n_claims"):
            continue  # nothing to annotate — an abstention has no claims
        for i in range(rec["n_claims"]):
            item_id = f"{rec['system']}::{rec['qid']}::c{i}"
            if item_id in store.items:
                continue
            store.add_item(
                AnnotationItem(
                    item_id=item_id,
                    qid=rec["qid"],
                    language=rec["language"],
                    question=rec.get("question", rec["qid"]),
                    claim=rec.get("claims", [None] * (i + 1))[i]
                    if rec.get("claims")
                    else rec["answer"],
                    evidence=[corpus.get(cid, cid) for cid in rec.get("retrieved_chunk_ids", [])],
                    system=rec["system"],
                )
            )
            added += 1

    print(f"loaded {added} new claims ({len(store.items)} total)")

    # ── judge selection: must differ from the system ─────────────────────────
    registry = load_registry()
    usable = [s.name for s in registry.usable() if s.name != "stub"]
    judge_name = args.judge or next((n for n in usable if n != args.system_provider), None)

    if judge_name is None:
        print(
            f"\nNo judge provider available that differs from the system "
            f"({args.system_provider!r}).\n"
            f"Configured: {usable or 'none'}\n\n"
            "Using the same model to label its own output is circular — the resulting\n"
            "F1 would measure self-consistency, not correctness. Add a second key\n"
            "(e.g. GOOGLE_API_KEY alongside GROQ_API_KEY) and re-run."
        )
        return 1

    router = LLMRouter(providers=[judge_name])
    try:
        annotator = LLMAnnotator(
            provider=router,
            model=args.judge_model or "",
            system_provider=args.system_provider,
            # The router reports its own name, not the backend it dispatches to,
            # so the judge is named explicitly. Without this the conflict guard
            # silently passes — which is exactly what happened the first time.
            judge_provider=judge_name,
            annotator_id=f"llm:{judge_name}",
        )
    except JudgeConflictError as exc:
        print(f"\nREFUSED: {exc}\n")
        router.close()
        return 1

    pending = [i for i in store.items.values() if store.label(i.item_id, "llm") is None]
    print(f"judge: {bold(judge_name)} · labelling {len(pending)} claims\n")

    for n, item in enumerate(pending, 1):
        store.add_annotation(annotator.annotate(item))
        if n % 20 == 0:
            print(f"  {n}/{len(pending)}")

    router.close()
    print(f"\n{store.summary()}")
    print(f"\nNext: python scripts/annotate.py review --limit {args.review_target}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# review
# ──────────────────────────────────────────────────────────────────────────────


def cmd_review(args) -> int:
    """Human review of a stratified sample. This is the irreducible part."""
    store = AnnotationStore(args.store)
    if not store.items:
        print("store is empty — run `propose` first")
        return 1

    proposed = {i: store.label(i, "llm") or "UNKNOWN" for i in store.items}
    sample = stratified_sample(
        list(store.items.values()), proposed, fraction=args.fraction, seed=args.seed
    )
    todo = [i for i in sample if store.label(i.item_id, "human") is None]
    if args.language:
        todo = [i for i in todo if i.language in args.language]
    todo = todo[: args.limit]

    if not todo:
        print("nothing left to review in this stratum")
        print(store.summary())
        return 0

    print(f"\n{bold('Human review')} — {len(todo)} items")
    print("Judge label is HIDDEN until you decide, so it cannot anchor you.\n")
    print("  [s] SUPPORTED   [c] CONTRADICTED   [u] UNVERIFIABLE")
    print("  [?] show the judge's label    [n] skip    [q] save and quit\n")

    reviewed = 0
    for n, item in enumerate(todo, 1):
        print(rule(72))
        print(f"{n}/{len(todo)}  [{item.language}]  {item.item_id}")
        print(f"\n{bold('EVIDENCE')}")
        print(safe_text(_wrap(item.evidence_text or "(none retrieved)")))
        print(f"\n{bold('QUESTION')}  {safe_text(item.question)}")
        print(f"{bold('CLAIM')}     {safe_text(item.claim)}\n")

        revealed = False
        while True:
            choice = input("  label > ").strip().lower()
            if choice == "q":
                print(f"\nsaved. {store.summary()}")
                return 0
            if choice == "n":
                break
            if choice == "?":
                if not revealed:
                    print(f"  judge said: {store.label(item.item_id, 'llm')}")
                    revealed = True
                continue
            label = {"s": LABELS[0], "c": LABELS[1], "u": LABELS[2]}.get(choice)
            if label is None:
                print("  use s / c / u / ? / n / q")
                continue
            store.add_annotation(
                Annotation(
                    item_id=item.item_id,
                    label=label,
                    source="human",
                    annotator=args.annotator,
                    note="judge label revealed before deciding" if revealed else "",
                )
            )
            reviewed += 1
            break

    print(f"\n{rule(72)}\nreviewed {reviewed} items · {store.summary()}")
    print("\nNext: python scripts/annotate.py agreement")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# agreement
# ──────────────────────────────────────────────────────────────────────────────


def cmd_agreement(args) -> int:
    """Report judge-vs-human agreement. This is a result, not a diagnostic."""
    store = AnnotationStore(args.store)
    reports = store.agreement()

    if not reports:
        print("no items have both a judge label and a human label yet")
        return 1

    print(f"\n{bold('Judge vs. human agreement')}\n{rule(72)}")
    for lang, report in sorted(reports.items()):
        print(f"  {lang}  {report}")

    print(f"\n{bold('Per-label agreement')}\n{rule(72)}")
    for lang, report in sorted(reports.items()):
        detail = " · ".join(f"{k[:5]}={v:.2f}" for k, v in sorted(report.per_label_agreement.items()))
        print(f"  {lang}  {detail}")

    disagreements = store.disagreements()
    if disagreements:
        print(f"\n{bold('Disagreements')} — where the judge is unreliable\n{rule(72)}")
        for item, llm, human in disagreements[: args.show]:
            print(f"  [{item.language}] judge={llm:<13} human={human:<13} {safe_text(item.claim[:52])}")
        if len(disagreements) > args.show:
            print(f"  … and {len(disagreements) - args.show} more")

    weakest = min(reports.items(), key=lambda kv: kv[1].kappa)
    print(f"\n{rule(72)}")
    print(f"  Weakest agreement: {weakest[0]} at κ={weakest[1].kappa:.3f}")
    if not weakest[1].acceptable:
        print(
            "\n  κ < 0.60 — either the guidelines are ambiguous (revise and redo the\n"
            "  batch), or the judge is genuinely unreliable in this language. The\n"
            "  second is a FINDING and supports the project's thesis; report it."
        )
    else:
        print("\n  Report these figures alongside baseline #3 (LLM-as-judge): they")
        print("  give its reliability per language, not just its F1.")

    if args.out:
        payload = {
            lang: {
                "kappa": r.kappa,
                "raw_agreement": r.raw_agreement,
                "n": r.n,
                "interpretation": r.interpretation,
                "acceptable": r.acceptable,
                "per_label": r.per_label_agreement,
            }
            for lang, r in reports.items()
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  written to {args.out.relative_to(ROOT)}")
    print()
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# export
# ──────────────────────────────────────────────────────────────────────────────


def cmd_export(args) -> int:
    store = AnnotationStore(args.store)
    written = store.export(args.out)

    human = sum(1 for i in store.items if store.label(i, "human"))
    llm_only = written - human

    print(f"\nexported {written} labels to {args.out.relative_to(ROOT)}")
    print(f"  human-verified: {human}")
    print(f"  judge-only:     {llm_only}")

    if llm_only:
        by_lang: dict[str, int] = {}
        for item_id, item in store.items.items():
            if not store.label(item_id, "human") and store.label(item_id, "llm"):
                by_lang[item.language] = by_lang.get(item.language, 0) + 1
        print(f"  judge-only by language: {by_lang}")
        print(
            "\n  `label_source` is exported per item. Report results separately for\n"
            "  human-verified and judge-only labels — required wherever no human\n"
            "  reader was available (docs/03_PROPOSAL.md §7.1, risk R2)."
        )
    print()
    return 0


# ──────────────────────────────────────────────────────────────────────────────


def _load_corpus_text(corpus_dir: Path | None) -> dict[str, str]:
    """chunk_id -> text, so the annotator sees the evidence rather than an id."""
    from pramana.api.demo_corpus import DEMO_CORPUS
    from pramana.ingestion.chunking import chunk_text

    sources = (
        DEMO_CORPUS
        if corpus_dir is None
        else {
            lang: "\n\n".join(f.read_text(encoding="utf-8") for f in sorted((corpus_dir / lang).rglob("*.md")))
            for lang in ("en", "hi", "ta")
            if (corpus_dir / lang).exists()
        }
    )
    out: dict[str, str] = {}
    for lang, text in sources.items():
        prefix = "demo" if corpus_dir is None else "corpus"
        for c in chunk_text(text, doc_id=f"{prefix}_{lang}", language=lang):  # type: ignore[arg-type]
            out[c.chunk_id] = c.text
    return out


def _wrap(text: str, width: int = 72) -> str:
    import textwrap

    return "\n".join(
        textwrap.fill(line, width) for line in text.splitlines() if line.strip()
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose", help="judge labels every claim")
    p.add_argument("--raw", type=Path, required=True, help="reports/<run>/raw.jsonl")
    p.add_argument("--corpus", type=Path, help="corpus dir; omit for the demo corpus")
    p.add_argument("--system-provider", default="groq", help="provider the SYSTEM used")
    p.add_argument("--judge", help="provider for the judge; must differ from the system")
    p.add_argument("--judge-model", help="judge model id")
    p.add_argument("--review-target", type=int, default=40)
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("review", help="human review of a stratified sample")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--fraction", type=float, default=0.30)
    p.add_argument("--language", nargs="+", help="restrict to these languages")
    p.add_argument("--annotator", default="human1")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("agreement", help="judge-vs-human kappa, per language")
    p.add_argument("--show", type=int, default=15)
    p.add_argument("--out", type=Path, help="write JSON here")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("export", help="write final labels for the harness")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "annotations.jsonl")
    p.set_defaults(func=cmd_export)

    args = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

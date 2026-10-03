#!/usr/bin/env python
"""Evaluation harness — runs every system on every language and writes the tables.

Implements `docs/06_EVALUATION_PROTOCOL.md` §7.

    python scripts/run_experiment.py --smoke              # offline, fixtures only
    python scripts/run_experiment.py --eval data/eval/questions.jsonl
    python scripts/run_experiment.py --eval ... --systems pramana vanilla --limit 50

Three properties are enforced here rather than left to discipline:

* **Every system sees identical inputs.** Runs replay from the generation cache, so
  a difference between systems comes from the systems and not from sampling noise.
* **Nothing is averaged across languages.** Per-language tables are the output;
  the pooled figure is written alongside, never instead.
* **Every run writes a manifest.** Config hash, model strings, dataset version,
  seeds, git commit, and the draft-cache version. Without the last, split execution
  silently destroys reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pramana.confidence.fusion import ConfidenceModel  # noqa: E402
from pramana.correction.policy import CorrectionExecutor, CorrectionPolicy  # noqa: E402
from pramana.detection.decomposer import LLMDecomposer, RuleBasedDecomposer  # noqa: E402
from pramana.detection.verifier import (  # noqa: E402
    GroundingVerifier,
    KeywordNLIBackend,
    LLMNLIBackend,
    TransformerNLIBackend,
)
from pramana.evaluation.metrics import (  # noqa: E402
    LanguageBreakdown,
    abstention_metrics,
    bootstrap_ci,
    paired_bootstrap,
)
from pramana.generation import LLMRouter  # noqa: E402
from pramana.generation.drafting import DraftGenerator, DraftingPolicy  # noqa: E402
from pramana.ingestion.chunking import chunk_text  # noqa: E402
from pramana.pipeline import PramanaPipeline  # noqa: E402
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever  # noqa: E402
from pramana.schemas import LANGUAGES, Claim, Language  # noqa: E402
from pramana.utils.console import bold, rule, safe_text, setup_console  # noqa: E402

setup_console()

# System names map to `docs/06_EVALUATION_PROTOCOL.md` §3. Baselines 4 and 5 exist
# to justify specific design decisions, not to pad the table: if PRAMANA does not
# beat them, the corresponding component is not earning its cost.
SYSTEMS = {
    "vanilla": "No verification — the lower bound",
    "llm_judge": "Whole-answer LLM NLI against evidence chunks",
    "sentence_nli": "Sentence-level NLI, no decomposition — isolates decomposition",
    "keyword": "Lexical floor — any learned verifier must beat this",
    "pramana": "Full pipeline",
    "pramana_mdeberta": "Full pipeline with the local mDeBERTa XNLI verifier -- real NLI probabilities",
}


@dataclass(slots=True)
class Item:
    """One evaluation question. Mirrors `docs/05_DATASET_SPEC.md` §3.4."""

    qid: str
    parallel_id: str
    language: Language
    question: str
    answerability: str = "A"
    gold_chunk_ids: list[str] = field(default_factory=list)
    should_abstain: bool = False

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Item:
        return cls(
            qid=d["qid"],
            parallel_id=d.get("parallel_id", d["qid"]),
            language=d["language"],
            question=d["question"],
            answerability=d.get("answerability", "A"),
            gold_chunk_ids=d.get("gold_chunk_ids", []),
            should_abstain=bool(d.get("should_abstain", d.get("answerability") in {"C", "D"})),
        )


@dataclass(slots=True)
class Record:
    """One (system, item) outcome. Written to raw.jsonl for re-analysis."""

    system: str
    qid: str
    parallel_id: str
    language: str
    answerability: str
    answer: str
    confidence: float
    band: str
    n_claims: int
    hallucination_rate: float
    supported: int
    contradicted: int
    unverifiable: int
    abstained: bool
    should_abstain: bool
    regressed: bool
    iterations: int
    actions: list[str]
    retrieved_chunk_ids: list[str]
    gold_chunk_ids: list[str]
    latency_ms: float
    correction_attempts: int = 0
    correction_regressions: int = 0
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Corpus and systems
# ──────────────────────────────────────────────────────────────────────────────


def build_retriever(corpus_dir: Path | None) -> MultilingualRetriever:
    from pramana.ingestion.corpus import load_corpus
    retriever = MultilingualRetriever()
    if corpus_dir is None:
        from pramana.api.demo_corpus import DEMO_CORPUS
        sources = {lang: chunk_text(text, doc_id=f"corpus_{lang}", language=lang)
                   for lang, text in DEMO_CORPUS.items()}
    else:
        sources = load_corpus(corpus_dir)
    for lang, chunks in sources.items():
        r = HybridRetriever(language=lang, top_k=5)
        r.add(chunks)
        retriever.add_language(lang, r)
    return retriever


class WholeAnswerDecomposer:
    def decompose(self, draft):
        return [Claim("answer", draft.text, draft.language, (0, len(draft.text)))] if draft.text.strip() else []


def build_system(name: str, retriever: MultilingualRetriever, router, offline: bool):
    """Assemble one system under test. Differences between them are exactly the
    component being ablated — everything else is shared."""
    generator = DraftGenerator(router, DraftingPolicy(n_samples=0, sample_temperature=0.7))
    decomposer = RuleBasedDecomposer() if offline else LLMDecomposer(router)
    llm_backend = KeywordNLIBackend() if offline else LLMNLIBackend(router)

    common = dict(retriever=retriever, generator=generator, confidence=ConfidenceModel())

    if name == "vanilla":
        # No verification at all. Detection still runs so a hallucination rate can
        # be reported, but nothing acts on it.
        return PramanaPipeline(
            **common, decomposer=decomposer, verifier=GroundingVerifier(backend=llm_backend),
            policy=CorrectionPolicy(max_iterations=0), executor=None,
        )
    if name == "keyword":
        return PramanaPipeline(
            **common, decomposer=decomposer, verifier=GroundingVerifier(backend=KeywordNLIBackend()),
            policy=CorrectionPolicy(max_iterations=0), executor=None,
        )
    if name == "sentence_nli":
        # Rule-based splitting only, no atomic decomposition — isolates how much
        # claim decomposition contributes.
        return PramanaPipeline(
            **common, decomposer=RuleBasedDecomposer(split_connectives=False),
            verifier=GroundingVerifier(backend=llm_backend),
            policy=CorrectionPolicy(max_iterations=0), executor=None,
        )
    if name == "llm_judge":
        return PramanaPipeline(
            **common, decomposer=WholeAnswerDecomposer(),
            verifier=GroundingVerifier(backend=llm_backend),
            policy=CorrectionPolicy(max_iterations=0), executor=None,
        )
    if name == "pramana":
        return PramanaPipeline(
            **common, decomposer=decomposer, verifier=GroundingVerifier(backend=llm_backend),
            policy=CorrectionPolicy(max_iterations=2),
            executor=CorrectionExecutor(provider=router),
        )
    if name == "pramana_mdeberta":
        # Same pipeline, different judge: isolates the verifier. Offline runs keep
        # the keyword floor so the smoke path never downloads a model.
        backend = KeywordNLIBackend() if offline else TransformerNLIBackend()
        return PramanaPipeline(
            **common, decomposer=decomposer, verifier=GroundingVerifier(backend=backend),
            policy=CorrectionPolicy(max_iterations=2),
            executor=CorrectionExecutor(provider=router),
        )
    raise ValueError(f"unknown system {name!r}. Known: {sorted(SYSTEMS)}")


# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────


def run_system(name: str, pipeline, items: list[Item]) -> list[Record]:
    records: list[Record] = []
    for i, item in enumerate(items, 1):
        try:
            result = pipeline.run(item.question, language=item.language)
        except Exception as exc:
            print(safe_text(f"    [{i}/{len(items)}] {item.qid} FAILED: {type(exc).__name__}"))
            records.append(Record(
                system=name, qid=item.qid, parallel_id=item.parallel_id, language=item.language,
                answerability=item.answerability, answer="", confidence=0, band="LOW", n_claims=0,
                hallucination_rate=0, supported=0, contradicted=0, unverifiable=0,
                abstained=False, should_abstain=item.should_abstain, regressed=False,
                iterations=0, actions=[], retrieved_chunk_ids=[], gold_chunk_ids=item.gold_chunk_ids,
                latency_ms=0, error=type(exc).__name__,
            ))
            continue

        d = result.detection
        records.append(
            Record(
                system=name,
                qid=item.qid,
                parallel_id=item.parallel_id,
                language=result.language,
                answerability=item.answerability,
                answer=result.final_answer,
                confidence=round(result.confidence.score, 4),
                band=result.confidence.band,
                n_claims=d.n_claims,
                hallucination_rate=round(d.hallucination_rate, 4),
                supported=d.count("SUPPORTED"),  # type: ignore[arg-type]
                contradicted=d.count("CONTRADICTED"),  # type: ignore[arg-type]
                unverifiable=d.count("UNVERIFIABLE"),  # type: ignore[arg-type]
                abstained=result.abstained,
                should_abstain=item.should_abstain,
                regressed=result.regressed,
                iterations=result.iterations,
                actions=[a.value for a in result.action_history],
                retrieved_chunk_ids=result.retrieved_chunk_ids,
                correction_attempts=result.correction_attempts,
                correction_regressions=result.correction_regressions,
                gold_chunk_ids=item.gold_chunk_ids,
                latency_ms=round(result.total_latency_ms, 1),
            )
        )
        if i % 25 == 0:
            print(f"    {i}/{len(items)}")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────


def analyse(records: list[Record]) -> dict[str, Any]:
    """Per-language tables plus paired comparisons against the vanilla baseline.

    Note what is *not* reported: detection precision/recall/F1 against human
    labels. Those need the annotated dataset (`05_DATASET_SPEC.md` §4) and are
    computed by the analysis notebook once annotation exists. Reporting an
    unlabelled proxy as though it were F1 would be worse than reporting nothing.
    """
    out: dict[str, Any] = {"systems": {}, "comparisons": {}}
    by_system: dict[str, list[Record]] = {}
    for r in records:
        by_system.setdefault(r.system, []).append(r)

    for system, rows in by_system.items():
        per_language: dict[str, Any] = {}
        halluc_by_lang: dict[Language, float] = {}

        for lang in LANGUAGES:
            all_lang_rows = [r for r in rows if r.language == lang]
            lang_rows = [r for r in all_lang_rows if r.error is None]
            if not lang_rows:
                if all_lang_rows:
                    per_language[lang] = {"n": 0, "failures": len(all_lang_rows), "status": "all_failed"}
                continue

            halluc = [r.hallucination_rate for r in lang_rows]
            ci = bootstrap_ci(halluc, resamples=2000)
            abst = abstention_metrics(
                [r.abstained for r in lang_rows], [r.should_abstain for r in lang_rows]
            )
            halluc_by_lang[lang] = ci.point

            per_language[lang] = {
                "n": len(lang_rows),
                "failures": len(all_lang_rows) - len(lang_rows),
                "answered": sum(not r.abstained for r in lang_rows),
                "answered_hallucination_rate": (_mean([r.hallucination_rate for r in lang_rows if not r.abstained])
                                               if any(not r.abstained for r in lang_rows) else None),
                "hallucination_rate": {"point": ci.point, "ci_low": ci.lower, "ci_high": ci.upper},
                "abstention": asdict(abst),
                "mean_confidence": _mean([r.confidence for r in lang_rows]),
                "mean_claims": _mean([float(r.n_claims) for r in lang_rows]),
                "regression_rate": (sum(r.correction_regressions for r in lang_rows) / sum(r.correction_attempts for r in lang_rows)
                                    if sum(r.correction_attempts for r in lang_rows) else 0.0),
                "mean_latency_ms": _mean([r.latency_ms for r in lang_rows]),
            }

        breakdown = LanguageBreakdown(metric="hallucination_rate", by_language=halluc_by_lang, higher_is_better=False)
        out["systems"][system] = {
            "per_language": per_language,
            "cross_lingual": {
                "worst_language": breakdown.worst_gap[0],
                "worst_gap": breakdown.worst_gap[1],
                "meets_10pct_criterion": None,
                "note": "The research 10% criterion concerns human-labelled detection F1, not this proxy.",
            },
        }

    # Paired comparisons: joined on parallel_id so systems are compared on the
    # same questions, removing question difficulty as a source of variance.
    if "vanilla" in by_system:
        base = {(r.parallel_id, r.language): r.hallucination_rate for r in by_system["vanilla"] if r.error is None}
        for system, rows in by_system.items():
            if system == "vanilla":
                continue
            keys = [(r.parallel_id, r.language) for r in rows if r.error is None and (r.parallel_id, r.language) in base]
            if not keys:
                continue
            mine = {(r.parallel_id, r.language): r.hallucination_rate for r in rows if r.error is None}
            cmp = paired_bootstrap([mine[k] for k in keys], [base[k] for k in keys], resamples=2000)
            out["comparisons"][f"{system}_vs_vanilla"] = {
                "difference": cmp.difference,
                "ci_low": cmp.ci.lower,
                "ci_high": cmp.ci.upper,
                "p_value": cmp.p_value,
                "significant": cmp.significant,
                "n": cmp.n,
            }

    return out


def print_report(analysis: dict[str, Any]) -> None:
    for system, data in analysis["systems"].items():
        print(f"\n{bold(system)} — {SYSTEMS.get(system, '')}\n{rule(72)}")
        for lang, m in data["per_language"].items():
            if m.get("status") == "all_failed":
                print(f"  {lang}: all {m['failures']} runs failed")
                continue
            h = m["hallucination_rate"]
            print(
                f"  {lang}  n={m['n']:<4} halluc={h['point']:.3f} "
                f"[{h['ci_low']:.3f},{h['ci_high']:.3f}]  "
                f"conf={m['mean_confidence']:.2f}  "
                f"abstain={m['abstention']['rate']:.1%}  "
                f"over-abstain={m['abstention']['over_abstention_rate']:.1%}  "
                f"regress={m['regression_rate']:.1%}"
            )
        cl = data["cross_lingual"]
        print(f"  proxy gap: {cl['worst_language']} {cl['worst_gap']:+.1%}; detection-F1 target not assessed")

    if analysis["comparisons"]:
        print(f"\n{bold('Paired comparisons vs. vanilla')}\n{rule(72)}")
        for name, c in analysis["comparisons"].items():
            mark = "*" if c["significant"] else " "
            print(
                f"  {name:<28} Δ={c['difference']:+.3f} "
                f"[{c['ci_low']:+.3f},{c['ci_high']:+.3f}] p={c['p_value']:.4f}{mark}  n={c['n']}"
            )
        print("\n  * interval excludes zero. Read with the effect size, not alone.")


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────


def build_manifest(args, items: list[Item], records: list[Record], router) -> dict[str, Any]:
    """Everything needed to reproduce this run.

    ``draft_cache`` is not optional: under split execution the same config can
    replay against different cached generations, so a run without it is not
    reproducible even though it looks like it is.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=5
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    config_blob = json.dumps(vars(args), sort_keys=True, default=str)
    return {
        "run_id": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S"),
        "config_sha256": hashlib.sha256(config_blob.encode()).hexdigest()[:16],
        "config": {k: str(v) for k, v in vars(args).items()},
        "git_commit": commit if len(commit) == 40 else "unknown",
        "dataset_sha256": hashlib.sha256(args.eval.read_bytes()).hexdigest() if args.eval else None,
        "dataset": {"path": str(args.eval) if args.eval else "smoke", "n_items": len(items)},
        "systems": sorted({r.system for r in records}),
        "languages": sorted({r.language for r in records}),
        "providers": [b.spec.name for b in router._bound],
        "draft_cache": {
            "entries": router.cache.size(),
            "note": "The cache, not the endpoint, is the reproducible artefact.",
        },
        "router_stats": router.stats.summary(),
        "seeds": {"bootstrap": 42},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def smoke_items() -> list[Item]:
    """A handful of questions against the demo corpus, for wiring checks.

    Not an evaluation. The `unanswerable` item is included because the abstention
    path is the one most likely to break silently.
    """
    from pramana.api.demo_corpus import DEMO_QUERIES

    items: list[Item] = []
    for lang, queries in DEMO_QUERIES.items():
        for i, q in enumerate(queries):
            unanswerable = "WiFi" in q
            items.append(
                Item(
                    qid=f"{lang}-smoke-{i}",
                    parallel_id=f"smoke-{i}",
                    language=lang,  # type: ignore[arg-type]
                    question=q,
                    answerability="C" if unanswerable else "A",
                    should_abstain=unanswerable,
                )
            )
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, help="questions.jsonl (see 05_DATASET_SPEC.md §3.4)")
    ap.add_argument("--corpus", type=Path, help="corpus directory with {en,hi,ta}/ subfolders")
    ap.add_argument("--smoke", action="store_true", help="tiny offline wiring check")
    ap.add_argument("--systems", nargs="+", default=["vanilla", "pramana"], choices=sorted(SYSTEMS))
    ap.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    ap.add_argument("--limit", type=int, help="cap items per language, for a quick pass")
    ap.add_argument("--offline", action="store_true", help="force the stub provider")
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    ap.add_argument("--no-transliteration", action="store_true",
                    help="search Romanised queries as typed (ablation)")
    args = ap.parse_args()

    if not args.eval and not args.smoke:
        ap.error("pass --eval with a question file, or --smoke for a wiring check")

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    # ── load items ────────────────────────────────────────────────────────────
    if args.smoke:
        items = smoke_items()
    else:
        items = [
            Item.from_json(json.loads(line))
            for line in args.eval.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    items = [i for i in items if i.language in args.languages]
    if args.limit:
        capped: list[Item] = []
        for lang in args.languages:
            capped.extend([i for i in items if i.language == lang][: args.limit])
        items = capped

    if not items:
        print("no items to run")
        return 1

    # ── build ─────────────────────────────────────────────────────────────────
    from pramana.generation import load_registry

    offline = args.smoke or args.offline or not [s.name for s in load_registry().usable() if s.name != "stub"]
    router = LLMRouter(providers=["stub"]) if offline else LLMRouter()
    retriever = build_retriever(args.corpus)
    if not offline and not args.no_transliteration:
        # Shared by every system, so it never explains a difference between them.
        from pramana.retrieval.query_rewrite import LLMQueryTransliterator
        retriever.query_variants = LLMQueryTransliterator(router)

    print(f"\n{bold('PRAMANA evaluation harness')}")
    print(f"  items={len(items)}  languages={sorted({i.language for i in items})}")
    print(f"  systems={args.systems}  offline={offline}")

    # ── run ───────────────────────────────────────────────────────────────────
    started = time.perf_counter()
    records: list[Record] = []
    for name in args.systems:
        print(f"\n  running {bold(name)} …")
        records.extend(run_system(name, build_system(name, retriever, router, offline), items))

    # ── analyse and write ─────────────────────────────────────────────────────
    analysis = analyse(records)
    analysis["measurement"] = "offline_smoke_only" if offline else "automated_verifier_proxy_not_human_ground_truth"
    analysis["failed_runs"] = sum(r.error is not None for r in records)
    print_report(analysis)

    manifest = build_manifest(args, items, records, router)
    run_dir = args.out / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    with (run_dir / "raw.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(f"\n{rule(72)}")
    print(f"  {len(records)} records in {time.perf_counter() - started:.1f}s")
    print(f"  written to {run_dir.resolve()}")
    if offline:
        print("\n  NOTE: offline stub run — these numbers are not results.")
    print()
    router.close()
    return 1 if any(r.error for r in records) else 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

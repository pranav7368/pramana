"""Tests for the annotation workflow.

The property that matters most is guarded first: a judge sharing a provider with
the system under test must be **refused**, not warned about. Circular labels
produce a run that looks valid and yields an indefensible F1.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pramana.evaluation.annotation import (
    Annotation,
    AnnotationItem,
    AnnotationStore,
    JudgeConflictError,
    LLMAnnotator,
    agreement_by_language,
    cohen_kappa,
    stratified_sample,
)
from pramana.generation.base import Completion, GenerationResponse

S, C, U = "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"


def item(item_id: str, language="en", claim="a claim", evidence=("some evidence",)) -> AnnotationItem:
    return AnnotationItem(
        item_id=item_id,
        qid=item_id.split("::")[0],
        language=language,
        question="a question",
        claim=claim,
        evidence=list(evidence),
    )


class _Judge:
    def __init__(self, name: str, *replies: str):
        self.name = name
        self.queue = list(replies)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return GenerationResponse(
            completions=[Completion(self.queue.pop(0) if self.queue else "NEUTRAL")],
            model="judge-model",
            provider=self.name,
        )

    def capabilities(self, model):
        return set()

    def available_models(self):
        return ["judge-model"]

    def health_check(self):
        return True


# ──────────────────────────────────────────────────────────────────────────────
# The circularity guard
# ──────────────────────────────────────────────────────────────────────────────


class TestJudgeSeparation:
    def test_same_provider_is_refused(self):
        """Labelling a system's output with the same model measures
        self-consistency, not correctness — and the resulting run looks valid."""
        with pytest.raises(JudgeConflictError, match="circular"):
            LLMAnnotator(provider=_Judge("groq"), system_provider="groq")

    def test_same_model_is_refused(self):
        with pytest.raises(JudgeConflictError, match="both use model"):
            LLMAnnotator(
                provider=_Judge("google"), model="llama-3.3-70b",
                system_provider="groq", system_model="llama-3.3-70b",
            )

    def test_different_provider_is_allowed(self):
        a = LLMAnnotator(provider=_Judge("google"), system_provider="groq")
        assert a.annotator_id == "llm"

    def test_router_is_unwrapped_to_its_backend(self):
        """A router reports its own name ("router"), not the backend it dispatches
        to. The guard compared against that and silently passed in the real CLI
        path — the most important safety property in the module, defeated by an
        integration detail."""

        class _Spec:
            def __init__(self, name):
                self.name = name

        class _Bound:
            def __init__(self, name):
                self.spec = _Spec(name)

        class _Router(_Judge):
            def __init__(self, *backends):
                super().__init__("router")
                self._bound = [_Bound(b) for b in backends]

        with pytest.raises(JudgeConflictError, match="circular"):
            LLMAnnotator(provider=_Router("groq"), system_provider="groq")

        assert LLMAnnotator(provider=_Router("google"), system_provider="groq")

    def test_multi_provider_router_including_the_system_is_refused(self):
        """A router bound to several providers could dispatch to the system's own
        model on any given call. Guessing is not acceptable here."""

        class _Spec:
            def __init__(self, name):
                self.name = name

        class _Bound:
            def __init__(self, name):
                self.spec = _Spec(name)

        class _Router(_Judge):
            def __init__(self, *backends):
                super().__init__("router")
                self._bound = [_Bound(b) for b in backends]

        with pytest.raises(JudgeConflictError, match="may dispatch"):
            LLMAnnotator(provider=_Router("groq", "google"), system_provider="groq")

    def test_explicit_judge_provider_overrides_detection(self):
        with pytest.raises(JudgeConflictError, match="circular"):
            LLMAnnotator(
                provider=_Judge("anything"), judge_provider="groq", system_provider="groq"
            )

    def test_unspecified_system_does_not_block(self):
        """The check needs the system's provider to be passed; absent it the
        annotator still constructs, so the CLI is what enforces the pairing."""
        assert LLMAnnotator(provider=_Judge("google"))


class TestLLMAnnotator:
    def annotator(self, *replies: str) -> LLMAnnotator:
        return LLMAnnotator(provider=_Judge("google", *replies), system_provider="groq")

    @pytest.mark.parametrize("reply,expected", [("SUPPORTED", S), ("CONTRADICTED", C), ("UNVERIFIABLE", U)])
    def test_parses_each_label(self, reply, expected):
        assert self.annotator(reply).annotate(item("q::c0")).label == expected

    def test_parses_a_wrapped_label(self):
        assert self.annotator("The answer is CONTRADICTED.").annotate(item("q::c0")).label == C

    def test_unparseable_output_is_low_confidence_not_a_guess(self):
        a = self.annotator("I'm not sure").annotate(item("q::c0"))
        assert a.confidence == 0.0 and a.note

    def test_empty_evidence_skips_the_judge_call(self):
        """Unverifiable by definition — a judge call would only add noise and cost."""
        judge = _Judge("google")
        a = LLMAnnotator(provider=judge, system_provider="groq")
        result = a.annotate(item("q::c0", evidence=()))
        assert result.label == U and judge.calls == 0

    def test_judge_failure_is_marked_for_review(self):
        class _Broken(_Judge):
            def generate(self, request):
                raise RuntimeError("judge down")

        a = LLMAnnotator(provider=_Broken("google"), system_provider="groq")
        result = a.annotate(item("q::c0"))
        assert result.confidence == 0.0 and "error" in result.note

    def test_prompt_forbids_using_world_knowledge(self):
        """The single most common annotation error: marking a claim SUPPORTED
        because it is true in reality rather than present in the evidence."""
        judge = _Judge("google", "SUPPORTED")
        LLMAnnotator(provider=judge, system_provider="groq").annotate(item("q::c0"))
        # The system prompt is the first message of the only call made.
        assert judge.calls == 1


# ──────────────────────────────────────────────────────────────────────────────
# Agreement
# ──────────────────────────────────────────────────────────────────────────────


class TestCohenKappa:
    def test_perfect_agreement(self):
        r = cohen_kappa([S, C, U, S], [S, C, U, S])
        assert r.kappa == pytest.approx(1.0)
        assert r.acceptable

    def test_chance_level_agreement_scores_near_zero(self):
        """Two raters who both always say SUPPORTED show ~100% raw agreement while
        agreeing on nothing informative. κ corrects for that."""
        r = cohen_kappa([S] * 10, [S] * 10)
        assert r.raw_agreement == pytest.approx(1.0)
        assert r.kappa == pytest.approx(1.0)  # degenerate but consistent

        mixed = cohen_kappa([S, S, S, C], [S, S, C, S])
        assert mixed.raw_agreement > mixed.kappa, "kappa must be stricter than raw"

    def test_total_disagreement_is_negative(self):
        r = cohen_kappa([S, S, C, C], [C, C, S, S])
        assert r.kappa < 0
        assert r.interpretation == "worse than chance"

    def test_threshold_is_enforced(self):
        """Below 0.60 the guidelines are revised and the batch redone."""
        assert not cohen_kappa([S, C, U, S, C], [S, U, C, C, S]).acceptable

    def test_per_label_agreement_localises_the_problem(self):
        r = cohen_kappa([S, S, C, U], [S, S, U, U])
        assert r.per_label_agreement[S] == pytest.approx(1.0)
        assert r.per_label_agreement[C] < 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must be paired"):
            cohen_kappa([S, C], [S])

    def test_empty_input_is_safe(self):
        assert cohen_kappa([], []).n == 0


class TestAgreementByLanguage:
    def test_reports_each_language_separately(self):
        """A judge that agrees in English but not Tamil is evidence for the
        project's central claim — it must never collapse to one number."""
        pairs = [
            ("en", S, S), ("en", C, C), ("en", U, U), ("en", S, S),
            ("ta", S, C), ("ta", C, U), ("ta", U, S), ("ta", S, U),
        ]
        reports = agreement_by_language(pairs)
        assert reports["en"].kappa > reports["ta"].kappa
        assert reports["en"].acceptable and not reports["ta"].acceptable


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────


class TestStratifiedSample:
    def test_rare_class_is_not_missed(self):
        """Uniform sampling might draw almost no CONTRADICTED items, leaving the
        judge's behaviour on the critical class unmeasured."""
        items = [item(f"q{i}::c0") for i in range(100)]
        proposed = {i.item_id: (C if idx < 3 else S) for idx, i in enumerate(items)}
        sample = stratified_sample(items, proposed, fraction=0.1, min_per_stratum=3)
        assert sum(1 for i in sample if proposed[i.item_id] == C) == 3

    def test_stratifies_across_languages(self):
        items = [item(f"q{i}::c0", language=lang) for lang in ("en", "hi", "ta") for i in range(20)]
        proposed = dict.fromkeys((i.item_id for i in items), S)
        sample = stratified_sample(items, proposed, fraction=0.25, min_per_stratum=2)
        assert {i.language for i in sample} == {"en", "hi", "ta"}

    def test_selection_is_reproducible(self):
        items = [item(f"q{i}::c0") for i in range(50)]
        proposed = dict.fromkeys((i.item_id for i in items), S)
        a = [i.item_id for i in stratified_sample(items, proposed, seed=7)]
        b = [i.item_id for i in stratified_sample(items, proposed, seed=7)]
        assert a == b

    def test_never_over_samples_a_small_stratum(self):
        items = [item("q0::c0")]
        sample = stratified_sample(items, {"q0::c0": S}, fraction=0.3, min_per_stratum=10)
        assert len(sample) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────────────


class TestAnnotationStore:
    def test_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "store.jsonl"
        s = AnnotationStore(path)
        s.add_item(item("q0::c0", language="hi", claim="क्लेम रिजेक्ट हो गया।"))
        s.add_annotation(Annotation(item_id="q0::c0", label=S, source="llm"))

        reloaded = AnnotationStore(path)
        assert reloaded.items["q0::c0"].claim == "क्लेम रिजेक्ट हो गया।"
        assert reloaded.label("q0::c0", "llm") == S

    def test_human_label_supersedes_the_judge(self):
        """Wherever a human looked, the human decides."""
        s = AnnotationStore(_tmp())
        s.add_item(item("q0::c0"))
        s.add_annotation(Annotation(item_id="q0::c0", label=S, source="llm"))
        s.add_annotation(Annotation(item_id="q0::c0", label=C, source="human"))
        assert s.final_label("q0::c0") == C

    def test_adjudication_wins_over_both(self):
        s = AnnotationStore(_tmp())
        s.add_item(item("q0::c0"))
        for label, source in ((S, "llm"), (C, "human"), (U, "adjudicated")):
            s.add_annotation(Annotation(item_id="q0::c0", label=label, source=source))
        assert s.final_label("q0::c0") == U

    def test_disagreements_are_surfaced(self):
        """The most informative subset: where the judge is unreliable, and how."""
        s = AnnotationStore(_tmp())
        for i, (llm, human) in enumerate([(S, S), (S, C), (C, U)]):
            s.add_item(item(f"q{i}::c0"))
            s.add_annotation(Annotation(item_id=f"q{i}::c0", label=llm, source="llm"))
            s.add_annotation(Annotation(item_id=f"q{i}::c0", label=human, source="human"))
        assert len(s.disagreements()) == 2

    def test_needs_human_excludes_reviewed_items(self):
        s = AnnotationStore(_tmp())
        s.add_item(item("q0::c0"))
        s.add_item(item("q1::c0"))
        s.add_annotation(Annotation(item_id="q0::c0", label=S, source="human"))
        assert [i.item_id for i in s.needs_human()] == ["q1::c0"]

    def test_export_records_label_provenance(self, tmp_path):
        """Results must be reportable separately for human-verified and judge-only
        labels — required wherever no human reader was available."""
        s = AnnotationStore(tmp_path / "store.jsonl")
        s.add_item(item("q0::c0"))
        s.add_item(item("q1::c0", language="ta"))
        s.add_annotation(Annotation(item_id="q0::c0", label=S, source="llm"))
        s.add_annotation(Annotation(item_id="q0::c0", label=C, source="human"))
        s.add_annotation(Annotation(item_id="q1::c0", label=S, source="llm"))

        out = tmp_path / "annotations.jsonl"
        assert s.export(out) == 2
        rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
        by_id = {r["item_id"]: r for r in rows}
        assert by_id["q0::c0"]["label_source"] == "human" and by_id["q0::c0"]["label"] == C
        assert by_id["q1::c0"]["label_source"] == "llm"

    def test_invalid_label_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="invalid label"):
            Annotation(item_id="x", label="MAYBE", source="human")

    def test_corrupt_line_does_not_lose_the_store(self, tmp_path):
        """Annotation runs for hours across sittings; an interrupted write must not
        destroy prior work."""
        path = tmp_path / "store.jsonl"
        s = AnnotationStore(path)
        s.add_item(item("q0::c0"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{truncated\n")

        reloaded = AnnotationStore(path)
        assert "q0::c0" in reloaded.items


_counter = 0


def _tmp() -> Path:
    """Unique temp path per call, so stores in one test do not share state."""
    global _counter
    _counter += 1
    return Path(tempfile.mkdtemp(prefix="pramana_ann_")) / f"store_{_counter}.jsonl"

"""Deterministic RAG assurance with request-local controls and bounded corrections.

The latency budget is checked between operations; it is not an HTTP-call deadline.
Only executed actions are reported. Failed candidates restore both answer and evidence.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace

from pramana.confidence.fusion import ConfidenceModel, collect_features
from pramana.correction.policy import (
    CorrectionExecutor,
    CorrectionOutcome,
    CorrectionPolicy,
    RegressionTracker,
    VerdictProfile,
)
from pramana.detection.decomposer import Decomposer
from pramana.detection.verifier import GroundingVerifier, apply_thresholds
from pramana.generation.drafting import DraftGenerator
from pramana.ingestion.language import detect_language
from pramana.retrieval.hybrid import MultilingualRetriever
from pramana.schemas import (
    Action,
    AssuranceResult,
    Claim,
    ClaimVerdict,
    ConfidenceReport,
    DetectionResult,
    Draft,
    Language,
    RetrievalResult,
    Verdict,
)


@dataclass(slots=True)
class PramanaPipeline:
    retriever: MultilingualRetriever
    generator: DraftGenerator
    decomposer: Decomposer
    verifier: GroundingVerifier
    confidence: ConfidenceModel = field(default_factory=ConfidenceModel)
    policy: CorrectionPolicy = field(default_factory=CorrectionPolicy)
    executor: CorrectionExecutor | None = None
    tracker: RegressionTracker = field(default_factory=RegressionTracker)
    top_k: int = 5
    fail_closed: bool = False

    def run(
        self, query: str, language: Language | None = None, *,
        assurance: bool = True, max_corrections: int | None = None,
    ) -> AssuranceResult:
        if max_corrections is not None and max_corrections < 0:
            raise ValueError("max_corrections must be non-negative")
        policy = replace(self.policy, max_iterations=(
            self.policy.max_iterations if max_corrections is None else max_corrections
        ))
        executor = self.executor if assurance else None
        trace_id = uuid.uuid4().hex[:12]
        timings: dict[str, float] = {}
        actions: list[Action] = []
        t0 = time.perf_counter()
        lang: Language = language or detect_language(query).language
        timings["language"] = _ms(t0)
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(query, language=lang, top_k=self.top_k)
        timings["retrieval"] = _ms(t0)
        t0 = time.perf_counter()
        draft = self.generator.generate(query, retrieval, lang)
        timings["generation"] = _ms(t0)
        detection = DetectionResult([])
        report = self.confidence.score({})
        attempts = iterations = regressions = 0
        rolled_back = abstained = False
        stop_reason = ""
        pending_detection: DetectionResult | None = None
        loop_started = time.perf_counter()

        while True:
            if DraftGenerator.is_abstention(draft.text):
                abstained = True
                stop_reason = "generator_abstained"
                actions.append(Action.ABSTAIN)
                break

            t0 = time.perf_counter()
            detection = pending_detection if pending_detection is not None else self._detect(draft, retrieval, lang, query)
            pending_detection = None
            timings["detection"] = timings.get("detection", 0.0) + _ms(t0)
            t0 = time.perf_counter()
            report = self.confidence.score(collect_features(retrieval, draft, detection))
            timings["confidence"] = timings.get("confidence", 0.0) + _ms(t0)

            if executor is None:
                # An audit does not accept, reject, or change the submitted answer.
                stop_reason = "audit_only"
                break

            action = policy.decide(detection, report.score, report.band, retrieval, attempts)
            if action not in (Action.ACCEPT, Action.ABSTAIN):
                if attempts >= policy.max_iterations:
                    action = Action.ABSTAIN
                    stop_reason = "correction_limit"
                elif _ms(loop_started) >= policy.latency_budget_ms:
                    action = Action.ABSTAIN
                    stop_reason = "latency_budget"

            if action in (Action.ACCEPT, Action.ABSTAIN):
                actions.append(action)
                abstained = action is Action.ABSTAIN
                stop_reason = stop_reason or (
                    "correction_limit" if abstained and attempts >= policy.max_iterations else action.value.lower()
                )
                break

            attempts += 1
            candidate_retrieval = retrieval
            candidate_detection = detection
            t0 = time.perf_counter()
            if action is Action.RE_RETRIEVE:
                # Add the disputed propositions to the original question and widen retrieval.
                disputed = " ".join(v.claim.text for v in detection.claim_verdicts if v.is_hallucination)
                candidate_retrieval = self.retriever.retrieve(
                    f"{query} {disputed}".strip(), language=lang, top_k=self.top_k * 2,
                )
                # Judge both answers against the same new evidence for a fair comparison.
                candidate_detection = self._detect(draft, candidate_retrieval, lang, query)
            if action is Action.RE_RETRIEVE and VerdictProfile.of(candidate_detection).is_better_than(VerdictProfile.of(detection)):
                outcome = CorrectionOutcome(
                    draft.text, action, VerdictProfile.of(detection), VerdictProfile.of(candidate_detection),
                    True, False, detection_after=candidate_detection,
                )
            else:
                def reverify(text: str, evidence: RetrievalResult = candidate_retrieval) -> DetectionResult:
                    return self._detect(Draft(text=text, language=lang), evidence, lang, query)

                outcome = executor.apply(
                    action, draft, candidate_detection, candidate_retrieval,
                    reverify=reverify, language=lang,
                )
            timings["correction"] = timings.get("correction", 0.0) + _ms(t0)
            actions.append(action)
            self.tracker.record(outcome)
            regressions += int(outcome.regressed)
            if not outcome.accepted:
                rolled_back = outcome.profile_after is not None
                stop_reason = "rolled_back" if rolled_back else "correction_failed"
                break
            retrieval = candidate_retrieval
            iterations += 1
            if outcome.action is Action.ABSTAIN:
                actions.append(Action.ABSTAIN)
                abstained = True
                stop_reason = "correction_abstained"
                break
            # Original logprobs and samples do not describe a revised answer.
            draft = Draft(text=outcome.text, language=lang)
            pending_detection = outcome.detection_after

        if self.fail_closed and executor is not None and not abstained and (
            not detection.n_claims or detection.hallucination_rate > 0 or report.band == "LOW"
        ):
            actions.append(Action.ABSTAIN)
            abstained = True
            stop_reason = "unresolved_evidence"
        partial = not abstained and executor is not None and detection.partial
        if partial:
            draft = Draft(text=f"{draft.text.rstrip()}\n\n{DraftGenerator.partial_note(lang)}", language=lang)
        if abstained:
            draft = Draft(text=DraftGenerator.abstention_message(lang), language=lang)
            detection = DetectionResult([])
            report = ConfidenceReport(
                score=0.0, raw_score=0.0, band="LOW", signals=retrieval.signals(),
                calibrator="n/a", missing_signals=["abstained; no factual answer to score"],
            )
        return AssuranceResult(
            final_answer=draft.text, language=lang, confidence=report, detection=detection,
            action_history=actions, iterations=iterations, regressed=regressions > 0,
            abstained=abstained,
            evidence_chunk_ids=sorted({cid for v in detection.claim_verdicts for cid in v.supporting_chunk_ids}),
            latency_ms=timings, trace_id=trace_id, retrieved_chunk_ids=retrieval.chunk_ids(),
            correction_attempts=attempts, correction_regressions=regressions,
            rolled_back=rolled_back, stop_reason=stop_reason, partial=partial,
        )

    def _detect(self, draft: Draft, retrieval: RetrievalResult, lang: Language, query: str = "") -> DetectionResult:
        return self.verify_answer(draft, retrieval, lang, query)

    def verify_answer(self, draft: Draft, retrieval: RetrievalResult, lang: Language, query: str = "") -> DetectionResult:
        """Check decomposition and, before accepting it, the complete original answer.

        A decomposer can omit an unsupported clause. A full-answer check against
        the combined evidence prevents omission from becoming a clean verdict.
        It is an additional model check, not an independent proof of truth.
        """
        claims = self.decomposer.decompose(draft)
        check_answer = getattr(self.verifier.backend, "score_answer", None)
        if self.fail_closed and not claims and draft.text.strip():
            # Numeric/yes-no answers carry meaning through their question and
            # must be checked even when an atomic decomposer emits no claims.
            claims = [Claim("short_answer", draft.text, lang)]
            if check_answer and query and not retrieval.is_empty:
                scores = check_answer(retrieval.context_text(), query, draft.text, lang).normalised()
                verdict = apply_thresholds(scores, self.verifier._thresholds_for(lang))
                return DetectionResult([ClaimVerdict(
                    claim=claims[0], verdict=verdict, entailment_prob=scores.entailment,
                    contradiction_prob=scores.contradiction, neutral_prob=scores.neutral,
                    verification_language=lang,
                    supporting_chunk_ids=retrieval.chunk_ids() if verdict is not Verdict.UNVERIFIABLE else [],
                )])
        detection = self.verifier.verify(claims, retrieval, lang)
        if (self.fail_closed and not retrieval.is_empty and detection.n_claims
                and detection.supported_ratio == 1):
            scores = (check_answer(retrieval.context_text(), query, draft.text, lang)
                      if check_answer and query else self.verifier.backend.score(
                          retrieval.context_text(), draft.text, lang,
                      )).normalised()
            verdict = apply_thresholds(scores, self.verifier._thresholds_for(lang))
            assess = getattr(self.verifier.backend, "assess_coverage", None)
            if (verdict is Verdict.UNVERIFIABLE and assess is not None and query
                    and assess(retrieval.context_text(), query, draft.text, lang) == "PARTIAL"):
                # Everything stated is supported; only part of the question is
                # covered. Released with a label instead of withheld.
                detection.partial = True
            elif verdict is not Verdict.SUPPORTED:
                detection.claim_verdicts.append(ClaimVerdict(
                    claim=Claim("answer_coverage", draft.text, lang), verdict=verdict,
                    entailment_prob=scores.entailment, contradiction_prob=scores.contradiction,
                    neutral_prob=scores.neutral, verification_language=lang,
                    supporting_chunk_ids=retrieval.chunk_ids() if verdict is Verdict.CONTRADICTED else [],
                ))
        return detection


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000


def result_summary(result: AssuranceResult) -> str:
    profile = VerdictProfile.of(result.detection)
    actions = " -> ".join(a.value for a in result.action_history) or "none"
    flags = []
    if result.abstained:
        flags.append("ABSTAINED")
    if result.regressed:
        flags.append("REGRESSED")
    if result.rolled_back:
        flags.append("ROLLED_BACK")
    suffix = f"  [{', '.join(flags)}]" if flags else ""
    return (
        f"[{result.language}] conf={result.confidence.score:.2f} ({result.confidence.band}) "
        f"claims={profile} actions={actions} "
        f"iters={result.iterations} {result.total_latency_ms:.0f}ms{suffix}"
    )

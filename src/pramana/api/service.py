"""FastAPI service exposing the assurance layer.

Deliverable **D6**. Two audiences:

* A **demo** that makes the contribution visible -- per-claim verdicts, evidence
  citations, the correction trail. A confidence number alone is not convincing;
  seeing *which claim* was flagged and *which chunk* refuted it is.
* An **operator surface**: ``/v1/verify`` audits answers a system already produced,
  which is how this would first be adopted in practice -- alongside an existing
  chatbot rather than replacing it.

Run:  ``uvicorn pramana.api.service:app --reload``
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from starlette.concurrency import run_in_threadpool

from pramana.api.runtime import build_pipeline
from pramana.api.security import ServiceBoundary
from pramana.config.settings import Settings
from pramana.generation.budget import request_budget
from pramana.schemas import AssuranceResult, Language

log = logging.getLogger(__name__)

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover - optional extra
    raise RuntimeError(
        "The API needs FastAPI. Install with: pip install -e '.[serve]'"
    ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Wire contracts
# ──────────────────────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    language: Language | None = Field(
        default=None, description="Omit to auto-detect from the query."
    )
    assurance: bool = Field(default=True, description="Set false for the vanilla RAG baseline.")
    max_corrections: int = Field(default=2, ge=0, le=3)


class VerifyRequest(BaseModel):
    """Audit an answer some other system produced. No generation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=8000)
    language: Language | None = None


class ClaimOut(BaseModel):
    text: str
    verdict: str
    evidence: list[str]
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float


class ConfidenceOut(BaseModel):
    score: float
    band: str
    calibrator: str
    signals: dict[str, float]
    missing_signals: list[str]


class AskResponse(BaseModel):
    answer: str
    detected_language: str
    confidence: ConfidenceOut
    claims: list[ClaimOut]
    action_history: list[str]
    iterations: int
    regressed: bool
    abstained: bool
    evidence_chunk_ids: list[str]
    latency_ms: dict[str, float]
    trace_id: str
    hallucination_rate: float
    correction_attempts: int
    correction_regressions: int
    rolled_back: bool
    stop_reason: str
    retrieved_chunk_ids: list[str]


def to_response(result: AssuranceResult) -> AskResponse:
    return AskResponse(
        answer=result.final_answer,
        detected_language=result.language,
        confidence=ConfidenceOut(
            score=round(result.confidence.score, 4),
            band=result.confidence.band,
            calibrator=result.confidence.calibrator,
            signals={k: round(v, 4) for k, v in result.confidence.signals.items()},
            missing_signals=result.confidence.missing_signals,
        ),
        claims=[
            ClaimOut(
                text=v.claim.text,
                verdict=v.verdict.value,
                evidence=v.supporting_chunk_ids,
                entailment_prob=round(v.entailment_prob, 4),
                contradiction_prob=round(v.contradiction_prob, 4),
                neutral_prob=round(v.neutral_prob, 4),
            )
            for v in result.detection.claim_verdicts
        ],
        action_history=[a.value for a in result.action_history],
        iterations=result.iterations,
        regressed=result.regressed,
        abstained=result.abstained,
        evidence_chunk_ids=result.evidence_chunk_ids,
        latency_ms={k: round(v, 1) for k, v in result.latency_ms.items()},
        trace_id=result.trace_id,
        hallucination_rate=round(result.detection.hallucination_rate, 4),
        correction_attempts=result.correction_attempts,
        correction_regressions=result.correction_regressions,
        rolled_back=result.rolled_back, stop_reason=result.stop_reason,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Application state
# ──────────────────────────────────────────────────────────────────────────────

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    if settings.pilot:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    pipeline, router, offline = build_pipeline(settings=settings)
    _state.update(pipeline=pipeline, router=router, offline=offline, settings=settings,
                  slots=threading.BoundedSemaphore(settings.max_concurrent),
                  demo_original_retrievers=dict(pipeline.retriever.retrievers),
                  demo_documents={})
    log.info("PRAMANA ready mode=%s offline=%s", settings.mode, offline)
    try:
        yield
    finally:
        router.close()
        _state.clear()



app = FastAPI(
    title="PRAMANA",
    description=(
        "Hallucination detection and self-correction for multilingual enterprise RAG. "
        "Every claim is verified against retrieved evidence and cited."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(ServiceBoundary, state=_state)


def inference_slot():
    slots = _state.get("slots")
    if slots is None or not slots.acquire(blocking=False):
        raise HTTPException(503, "Inference capacity is busy; retry later", headers={"Retry-After": "2"})
    try:
        yield
    finally:
        slots.release()


def request_language(query, language, pipeline):
    from pramana.ingestion.language import detect_language
    lang = language or detect_language(query).language
    if lang not in pipeline.retriever.languages:
        raise HTTPException(422, "No approved corpus for the requested language")
    return lang


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "offline": _state.get("offline", True),
        "languages": _state["pipeline"].retriever.languages if "pipeline" in _state else [],
        "note": (
            "Running on the offline stub provider — answers come from fixtures, not a model. "
            "Set GROQ_API_KEY or GOOGLE_API_KEY in .env for live generation."
            if _state.get("offline", True)
            else "Live provider configured."
        ),
    }


@app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(inference_slot)])
def ask(request: AskRequest) -> AskResponse:
    """Full pipeline: retrieve, generate, verify, score, correct."""
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "pipeline not initialised")

    cfg = _state["settings"]
    if cfg.pilot and not request.assurance:
        raise HTTPException(422, "Pilot answers require assurance; use /v1/verify for audits")
    lang = request_language(request.query, request.language, pipeline)
    if _state["offline"] and lang in _state["demo_documents"]:
        raise HTTPException(422, "Answering an uploaded document needs a live provider; offline audit is available")
    try:
        with request_budget(cfg.max_provider_calls, cfg.request_budget_s):
            result = pipeline.run(
                request.query, language=lang,
                assurance=request.assurance, max_corrections=min(request.max_corrections, cfg.max_corrections),
            )
    except Exception as exc:
        log.error("pipeline failed type=%s", type(exc).__name__)
        raise HTTPException(503, "Answer service unavailable; retry later") from None

    return to_response(result)


@app.post("/v1/verify", response_model=AskResponse, dependencies=[Depends(inference_slot)])
def verify(request: VerifyRequest) -> AskResponse:
    """Audit an answer produced elsewhere. Retrieval and verification only.

    The realistic adoption path: point this at an existing chatbot's logs to
    measure its hallucination rate without changing it.
    """
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "pipeline not initialised")

    from pramana.confidence.fusion import collect_features
    from pramana.schemas import Draft

    lang = request_language(request.query, request.language, pipeline)
    started = time.perf_counter()
    draft = Draft(text=request.answer, language=lang)
    cfg = _state["settings"]
    try:
        with request_budget(cfg.max_provider_calls, cfg.request_budget_s):
            retrieval = pipeline.retriever.retrieve(request.query, language=lang)
            detection = pipeline.verify_answer(draft, retrieval, lang, request.query)
    except Exception as exc:
        log.error("verification failed type=%s", type(exc).__name__)
        raise HTTPException(503, "Verification service unavailable; retry later") from None
    report = pipeline.confidence.score(collect_features(retrieval, draft, detection))

    return to_response(
        AssuranceResult(
            final_answer=request.answer,
            language=lang,
            confidence=report,
            detection=detection,
            action_history=[],
            stop_reason="audit_only",
            retrieved_chunk_ids=retrieval.chunk_ids(),
            latency_ms={"verification": (time.perf_counter() - started) * 1000},
            evidence_chunk_ids=sorted(
                {c for v in detection.claim_verdicts for c in v.supporting_chunk_ids}
            ),
            trace_id=uuid.uuid4().hex[:12],
        )
    )


@app.get("/v1/corpus")
def corpus() -> dict[str, list[dict[str, Any]]]:
    """The indexed chunks. The demo shows these so a citation can be checked."""
    if not _state["settings"].expose_corpus:
        raise HTTPException(404, "Corpus browsing is disabled")
    pipeline = _state["pipeline"]
    return {
        lang: [
            {"chunk_id": c.chunk_id, "text": c.text,
             "source": c.metadata.get("source", "Sample policy"),
             "page": c.metadata.get("page"), "sha256": c.metadata.get("sha256")}
            for c in pipeline.retriever.retrievers[lang].chunks
        ]
        for lang in pipeline.retriever.languages
    }


def _require_demo() -> None:
    if _state["settings"].pilot:
        raise HTTPException(404, "Demo document upload is unavailable in pilot mode")


@app.get("/v1/demo/documents")
def demo_documents() -> dict[str, Any]:
    """Show which language indexes use a faculty upload versus sample policies."""
    _require_demo()
    pipeline = _state["pipeline"]
    return {
        "offline": _state["offline"],
        "provider": "stub" if _state["offline"] else ", ".join(_state["settings"].providers),
        "confidence_fitted": bool(pipeline.confidence.feature_names),
        "api_embedding_model": _state["settings"].api_embedding_model,
        "documents": {
            lang: _state["demo_documents"][lang].summary() if lang in _state["demo_documents"]
            else {"name": "Built-in fictional policy", "language": lang,
                  "pages": None, "chunks": retriever.size, "sample": True}
            for lang, retriever in pipeline.retriever.retrievers.items()
        },
    }


@app.post("/v1/demo/documents", dependencies=[Depends(inference_slot)])
async def upload_demo_document(
    request: Request, filename: str, language: Language = "en",
) -> dict[str, Any]:
    """Replace one language's demo index with an uploaded text PDF/Markdown/text file."""
    _require_demo()
    from pramana.api.demo_documents import DocumentError, prepare_document
    from pramana.retrieval.hybrid import HybridRetriever

    if language not in _state["pipeline"].retriever.languages:
        raise HTTPException(422, "This language is not enabled")
    try:
        prepared = await run_in_threadpool(prepare_document, await request.body(), filename, language)
    except DocumentError as exc:
        raise HTTPException(422, str(exc)) from None
    pipeline = _state["pipeline"]
    old = pipeline.retriever.retrievers[language]
    dense = None
    if old.dense is not None:
        from pramana.retrieval.dense import SentenceTransformerIndex
        dense = SentenceTransformerIndex(old.dense.model_name, encoder=old.dense.encoder)
    replacement = HybridRetriever(language=language, dense=dense, top_k=old.top_k,
                                  candidate_k=old.candidate_k)
    cfg = _state["settings"]
    try:
        with request_budget(cfg.max_provider_calls, cfg.request_budget_s):
            await run_in_threadpool(replacement.add, prepared.chunks)
    except Exception as exc:
        log.error("document indexing failed type=%s", type(exc).__name__)
        raise HTTPException(503, "Document indexing service unavailable; retry later") from None
    pipeline.retriever.add_language(language, replacement)
    _state["demo_documents"][language] = prepared
    return prepared.summary()


@app.delete("/v1/demo/documents", dependencies=[Depends(inference_slot)])
def reset_demo_documents(language: Language | None = None) -> dict[str, str]:
    """Restore one or all original fictional indexes without restarting the process."""
    _require_demo()
    pipeline = _state["pipeline"]
    originals = _state["demo_original_retrievers"]
    if language is not None and language not in originals:
        raise HTTPException(422, "This language is not enabled")
    for selected in (originals if language is None else (language,)):
        pipeline.retriever.add_language(selected, originals[selected])
        _state["demo_documents"].pop(selected, None)
    return {"status": "sample policies restored"}


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
    if _state["settings"].pilot:
        raise HTTPException(404, "Use the authenticated pilot API")
    from pramana.api.demo_ui import DEMO_HTML

    return DEMO_HTML


@app.get("/v1/ready")
def ready():
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Service is starting")
    return {
        "status": "ready", "mode": _state["settings"].mode,
        "offline": _state["offline"],
        "verification_backend": pipeline.verifier.backend.name,
        "confidence_fitted": bool(pipeline.confidence.feature_names),
        "retrieval": "hybrid" if (_state["settings"].dense_model or _state["settings"].api_embedding_model) else "sparse",
        "api_embedding_model": _state["settings"].api_embedding_model or None,
        "languages": pipeline.retriever.languages,
        "chunks": {lang: r.size for lang, r in pipeline.retriever.retrievers.items()},
        "note": "Readiness checks loaded components; it does not certify model accuracy or provider availability.",
    }


@app.get("/v1/runtime")
def runtime_status():
    """Operational counters without prompts, responses, credentials, or private paths.

    Protected by the same bearer-token boundary as other pilot routes.
    These are process counters; provider dashboards remain quota authority.
    """
    router = _state.get("router")
    if router is None:
        raise HTTPException(503, "Service is starting")
    stats = getattr(router, "stats", None)
    embedding_encoders = [getattr(r.dense, "encoder", None)
                          for r in _state["pipeline"].retriever.retrievers.values()]
    return {
        "mode": _state["settings"].mode,
        "offline": _state["offline"],
        "providers": [
            {"name": b.spec.name, "model": b.spec.default_model, "enabled": b.spec.enabled}
            for b in getattr(router, "_bound", [])
        ],
        "counters": asdict(stats) if stats else {},
        "embedding_api_requests": max((getattr(e, "requests", 0) for e in embedding_encoders), default=0),
        "note": "Process counters, not remaining account quota. Confidence is heuristic unless fitted.",
    }


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    from fastapi.responses import JSONResponse
    log.error("request=%s failed type=%s", getattr(request.state, "request_id", "unknown"), type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "Service unavailable; retry later"})

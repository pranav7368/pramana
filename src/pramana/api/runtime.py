"""Service assembly; no model calls or downloads occur at module import."""
from dataclasses import replace

from pramana.config.settings import Settings


def build_pipeline(offline: bool | None = None, *, settings: Settings | None = None):
    from pramana.api.demo_corpus import DEMO_CORPUS
    from pramana.confidence.fusion import ConfidenceModel
    from pramana.correction.policy import CorrectionExecutor, CorrectionPolicy
    from pramana.detection.decomposer import LLMDecomposer, RuleBasedDecomposer
    from pramana.detection.verifier import (
        GroundingVerifier,
        KeywordNLIBackend,
        LLMNLIBackend,
        TransformerNLIBackend,
    )
    from pramana.generation import LLMRouter, load_registry
    from pramana.generation.cache import GenerationCache
    from pramana.generation.drafting import DraftGenerator, DraftingPolicy
    from pramana.ingestion.chunking import chunk_text
    from pramana.ingestion.corpus import load_corpus
    from pramana.pipeline import PramanaPipeline
    from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever

    cfg = settings or Settings.from_env()
    registry = load_registry(cfg.provider_config)
    # Per-service copies prevent a failed request from modifying the global registry.
    registry = replace(registry, specs={name: replace(spec) for name, spec in registry.specs.items()})
    live = list(cfg.providers) or [s.name for s in registry.usable() if s.name != "stub"]
    if offline is None:
        offline = cfg.offline if cfg.offline is not None else not live
    if cfg.pilot and offline:
        raise ValueError("Pilot mode cannot use offline generation")
    selected = ["stub"] if offline else live
    for name in selected:
        spec = registry.get(name)
        if cfg.pilot and spec.adapter == "stub":
            raise ValueError("Pilot mode forbids stub adapters, including renamed providers")
        if not spec.enabled or not spec.is_configured:
            raise ValueError(f"Provider {name} is disabled or lacks credentials")
        spec.timeout_s = cfg.provider_timeout_s
        spec.max_retries = cfg.provider_retries
    if not selected:
        raise ValueError("No live provider configured")

    corpus = load_corpus(
        cfg.corpus_dir, cfg.languages, max_file_bytes=cfg.max_file_bytes,
        max_documents=cfg.max_documents, max_chunks=cfg.max_chunks,
    ) if cfg.corpus_dir else {
        lang: chunk_text(DEMO_CORPUS[lang], doc_id=f"demo_{lang}", language=lang)
        for lang in cfg.languages
    }
    confidence = ConfidenceModel.load(cfg.confidence_model) if cfg.confidence_model else ConfidenceModel()
    if cfg.confidence_model and not confidence.feature_names:
        raise ValueError("Configured confidence model is not fitted")
    if confidence.language and cfg.languages != (confidence.language,):
        raise ValueError("Language-specific confidence model requires matching PRAMANA_LANGUAGES")
    retriever = MultilingualRetriever()
    reranker = None
    if cfg.reranker_model:
        from pramana.retrieval.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker(cfg.reranker_model)
        reranker._ensure_loaded()  # Fail startup, not the first request, if unavailable.
    encoder = None
    if cfg.api_embedding_model:
        from pramana.retrieval.api_embeddings import GoogleEmbeddingEncoder
        encoder = GoogleEmbeddingEncoder(cfg.api_embedding_model, timeout=cfg.provider_timeout_s)
    for lang, chunks in corpus.items():
        dense = None
        if cfg.dense_model or cfg.api_embedding_model:
            from pramana.retrieval.dense import SentenceTransformerIndex
            dense = SentenceTransformerIndex(cfg.dense_model or cfg.api_embedding_model, encoder=encoder)
        r = HybridRetriever(language=lang, top_k=4, dense=dense, reranker=reranker)
        r.add(chunks)
        if dense is not None:
            encoder = dense.encoder  # Share one model across language indices.
        retriever.add_language(lang, r)

    router = LLMRouter(
        registry=registry, providers=selected,
        cache=GenerationCache(enabled=bool(cfg.cache_enabled)),
        rate_limit_wait_s=1.0 if cfg.pilot else 300.0,
    )
    if not offline and cfg.transliterate_queries:
        from pramana.retrieval.query_rewrite import LLMQueryTransliterator
        retriever.query_variants = LLMQueryTransliterator(router)
    try:
        backend = KeywordNLIBackend() if offline else (
            TransformerNLIBackend() if cfg.verifier == "transformer" else LLMNLIBackend(router, strict=True)
        )
        if isinstance(backend, TransformerNLIBackend):
            backend._ensure_loaded()  # Fail startup if required model is unavailable.
        pipeline = PramanaPipeline(
            retriever=retriever,
            generator=DraftGenerator(router, DraftingPolicy(n_samples=0, max_tokens=1024)),
            decomposer=RuleBasedDecomposer() if offline else LLMDecomposer(router, strict=True),
            verifier=GroundingVerifier(backend=backend, conflict_policy="contradiction" if not offline else "support"),
            confidence=confidence, policy=CorrectionPolicy(max_iterations=cfg.max_corrections),
            executor=CorrectionExecutor(provider=router, max_tokens=1024, strict=not offline), fail_closed=not offline,
        )
        return pipeline, router, offline
    except Exception:
        router.close()
        raise

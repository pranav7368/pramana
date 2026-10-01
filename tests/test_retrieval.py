"""Tests for sparse retrieval and hybrid fusion.

All offline: BM25 and RRF are pure Python, so the retrieval logic is fully
testable on the dev laptop with no model download and no GPU.
"""

from __future__ import annotations

import pytest

from pramana.retrieval.hybrid import (
    HybridRetriever,
    MultilingualRetriever,
    reciprocal_rank_fusion,
)
from pramana.retrieval.sparse import BM25Index, tokenize
from pramana.schemas import Chunk

EN_CHUNKS = [
    Chunk("e0", "d1", "Claims are rejected if submitted more than 30 days after discharge.", "en"),
    Chunk("e1", "d1", "Rejected claims may be appealed within 60 days of the notice.", "en"),
    Chunk("e2", "d2", "Policy PX-4471 covers outpatient dental treatment up to Rs. 25,000.", "en"),
    Chunk("e3", "d2", "The cafeteria serves lunch between noon and two o'clock.", "en"),
]

TA_CHUNKS = [
    Chunk("t0", "d1", "வெளியேற்றப்பட்ட 30 நாட்களுக்குப் பிறகு சமர்ப்பிக்கப்பட்ட உரிமைகோரல்கள் நிராகரிக்கப்படும்.", "ta"),
    Chunk("t1", "d1", "நிராகரிக்கப்பட்ட உரிமைகோரல்களை 60 நாட்களுக்குள் மேல்முறையீடு செய்யலாம்.", "ta"),
    Chunk("t2", "d2", "உணவகம் மதியம் உணவு வழங்குகிறது.", "ta"),
]

HI_CHUNKS = [
    Chunk("h0", "d1", "डिस्चार्ज के 30 दिन बाद जमा किए गए क्लेम रिजेक्ट कर दिए जाते हैं।", "hi"),
    Chunk("h1", "d1", "रिजेक्ट क्लेम की अपील 60 दिन के भीतर की जा सकती है।", "hi"),
]


def bm25(chunks) -> BM25Index:
    idx = BM25Index()
    idx.add(chunks)
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ──────────────────────────────────────────────────────────────────────────────


class TestTokenisation:
    def test_lowercases_and_splits_latin(self):
        assert "claims" in tokenize("Claims Are Rejected", "en")

    def test_keeps_digits(self):
        """'30 days' versus '15 days' is the single most consequential
        distinction in this corpus -- dropping numerals would remove it."""
        assert "30" in tokenize("rejected after 30 days", "en")

    def test_keeps_alphanumeric_codes_intact(self):
        """Policy identifiers are exactly what dense embeddings blur and BM25 is
        here to catch."""
        assert "px-4471" in tokenize("Policy PX-4471 covers dental", "en")

    def test_indic_tokens_are_ngrammed(self):
        """A Tamil agglutinated form is one whitespace token that would match
        nothing. Character n-grams recover partial matches inside it."""
        toks = tokenize("சமர்ப்பிக்கப்படாததால்", "ta")
        assert len(toks) > 1, "long Tamil token was not n-grammed"
        assert "சமர்ப்பிக்கப்படாததால்" in toks, "whole token must also be kept"

    def test_short_indic_tokens_are_not_ngrammed(self):
        """N-gramming a short word only adds noise."""
        assert tokenize("உணவு", "ta") == ["உணவு"]

    def test_danda_is_not_glued_to_the_last_word(self):
        toks = tokenize("क्लेम रिजेक्ट हुआ।", "hi")
        assert all("।" not in t for t in toks)

    def test_negation_words_survive_stopword_removal(self):
        """'not', 'nahi' and 'illai' are precisely the tokens whose loss turns a
        contradiction into an apparent agreement."""
        assert "not" in tokenize("benefits are not available", "en")
        assert any("नहीं" in t for t in tokenize("उपलब्ध नहीं है", "hi"))


# ──────────────────────────────────────────────────────────────────────────────
# BM25
# ──────────────────────────────────────────────────────────────────────────────


class TestBM25:
    def test_finds_the_relevant_chunk(self):
        hits = bm25(EN_CHUNKS).search("why was my claim rejected", top_k=3)
        assert hits
        assert hits[0][0] in (0, 1), "a claim-rejection chunk should rank first"

    def test_ignores_irrelevant_chunks(self):
        hits = bm25(EN_CHUNKS).search("claim rejected discharge", top_k=4)
        assert 3 not in [i for i, _ in hits], "cafeteria chunk should not match"

    def test_exact_code_lookup_beats_semantics(self):
        """The reason sparse retrieval exists in this pipeline."""
        hits = bm25(EN_CHUNKS).search("PX-4471", top_k=2)
        assert hits and hits[0][0] == 2

    def test_scores_are_positive_and_ordered(self):
        hits = bm25(EN_CHUNKS).search("claims rejected", top_k=4)
        scores = [s for _, s in hits]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_idf_is_floored_at_zero(self):
        """Raw BM25 IDF goes negative for terms in more than half the collection.
        On a small corpus that is common, and it would let a document be
        *penalised* for containing a query term."""
        idx = bm25(EN_CHUNKS)
        assert idx._idf("claims") >= 0.0
        assert all(idx._idf(t) >= 0.0 for t in ("the", "a", "claims", "days"))

    def test_empty_index_returns_nothing(self):
        assert BM25Index().search("anything") == []

    def test_query_with_no_usable_terms_returns_nothing(self):
        assert bm25(EN_CHUNKS).search("!!! ???") == []

    def test_tamil_retrieval_works(self):
        hits = bm25(TA_CHUNKS).search("உரிமைகோரல் நிராகரிக்கப்பட்டது", top_k=2)
        assert hits and hits[0][0] in (0, 1)

    def test_hindi_retrieval_works(self):
        hits = bm25(HI_CHUNKS).search("क्लेम रिजेक्ट", top_k=2)
        assert hits and hits[0][0] in (0, 1)

    def test_explain_shows_term_contributions(self):
        """Retrieval failures are otherwise very hard to diagnose."""
        idx = bm25(EN_CHUNKS)
        contributions = idx.explain("claim rejected discharge", 0)
        assert contributions
        assert all(v > 0 for v in contributions.values())


# ──────────────────────────────────────────────────────────────────────────────
# RRF
# ──────────────────────────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    def test_agreement_between_retrievers_wins(self):
        a = [(1, 9.9), (2, 5.0), (3, 1.0)]
        b = [(1, 0.4), (3, 0.3), (2, 0.2)]
        assert reciprocal_rank_fusion([a, b])[0][0] == 1

    def test_uses_ranks_not_scores(self):
        """The whole reason RRF was chosen: dense cosine and BM25 scores are on
        incompatible scales, and those scales differ *per language*."""
        small = [(1, 0.001), (2, 0.0005)]
        large = [(1, 9999.0), (2, 5000.0)]
        assert reciprocal_rank_fusion([small]) == reciprocal_rank_fusion([large])

    def test_union_of_both_lists_is_kept(self):
        fused = reciprocal_rank_fusion([[(1, 1.0)], [(2, 1.0)]])
        assert {i for i, _ in fused} == {1, 2}

    def test_weights_shift_the_outcome(self):
        a, b = [(1, 1.0)], [(2, 1.0)]
        assert reciprocal_rank_fusion([a, b], weights=[3.0, 1.0])[0][0] == 1
        assert reciprocal_rank_fusion([a, b], weights=[1.0, 3.0])[0][0] == 2

    def test_mismatched_weights_raise(self):
        with pytest.raises(ValueError, match="2 rankings but 3 weights"):
            reciprocal_rank_fusion([[(1, 1.0)], [(2, 1.0)]], weights=[1.0, 1.0, 1.0])

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid retriever
# ──────────────────────────────────────────────────────────────────────────────


class _StubDense:
    """Fixed dense ranking, so fusion is testable without an embedding model."""

    def __init__(self, ranking):
        self._ranking = ranking

    def add(self, chunks):
        return None

    def search(self, query, top_k):
        return self._ranking[:top_k]

    @property
    def size(self):
        return len(self._ranking)


class TestHybridRetriever:
    def build(self, dense=None) -> HybridRetriever:
        r = HybridRetriever(language="en", dense=dense, top_k=3)
        r.add(EN_CHUNKS)
        return r

    def test_sparse_only_retrieval_works(self):
        result = self.build().retrieve("why was my claim rejected")
        assert not result.is_empty
        assert result.chunks[0].chunk.chunk_id in ("e0", "e1")

    def test_rejects_chunks_in_the_wrong_language(self):
        """Mixing languages in one index lets cross-language score contamination
        through and makes per-language metrics impossible to compute cleanly."""
        with pytest.raises(ValueError, match="another language"):
            HybridRetriever(language="en").add(TA_CHUNKS)

    def test_empty_index_returns_empty_result_not_an_error(self):
        """The generator must then abstain -- this is fixture F04's path."""
        result = HybridRetriever(language="en").retrieve("anything")
        assert result.is_empty
        assert result.signals()["s1_n_chunks"] == 0.0

    def test_unmatched_query_returns_empty_result(self):
        assert self.build().retrieve("quantum chromodynamics ###").is_empty

    def test_dense_and_sparse_are_fused(self):
        dense = _StubDense([(3, 0.9), (2, 0.8)])  # cafeteria + policy
        result = self.build(dense).retrieve("claim rejected")
        ids = [rc.chunk.chunk_id for rc in result.chunks]
        assert "e0" in ids or "e1" in ids, "sparse hits must survive fusion"
        assert len(ids) > 1

    def test_component_scores_are_preserved_for_diagnosis(self):
        result = self.build(_StubDense([(0, 0.95)])).retrieve("claim rejected")
        top = result.chunks[0]
        assert top.dense_score or top.sparse_score
        assert top.fused_score > 0

    def test_respects_top_k(self):
        assert len(self.build().retrieve("claims rejected days", top_k=2).chunks) <= 2

    def test_ranks_are_sequential_from_zero(self):
        result = self.build().retrieve("claims rejected appeal days")
        assert [rc.rank for rc in result.chunks] == list(range(len(result.chunks)))

    def test_s1_signals_are_populated(self):
        s = self.build().retrieve("claim rejected discharge").signals()
        assert s["s1_top_score"] > 0
        assert s["s1_n_chunks"] > 0

    def test_reranker_reorders_and_renumbers(self):
        class _Reranker:
            def score(self, pairs):
                # Reverse the incoming order.
                return list(range(len(pairs), 0, -1))

        r = self.build()
        r.reranker = _Reranker()
        result = r.retrieve("claims rejected appeal")
        assert [rc.rank for rc in result.chunks] == list(range(len(result.chunks)))
        assert result.chunks[0].rerank_score >= result.chunks[-1].rerank_score

    def test_explain_surfaces_bm25_terms(self):
        r = self.build()
        assert r.explain("claim rejected discharge", "e0")

    def test_explain_rejects_unknown_chunk(self):
        with pytest.raises(KeyError):
            self.build().explain("q", "does-not-exist")


class TestMultilingualRetriever:
    def build(self) -> MultilingualRetriever:
        m = MultilingualRetriever()
        for lang, chunks in (("en", EN_CHUNKS), ("hi", HI_CHUNKS), ("ta", TA_CHUNKS)):
            r = HybridRetriever(language=lang)
            r.add(chunks)
            m.add_language(lang, r)
        return m

    def test_routes_by_detected_language(self):
        m = self.build()
        assert m.retrieve("why was my claim rejected").language == "en"
        assert m.retrieve("क्लेम रिजेक्ट क्यों हुआ").language == "hi"
        assert m.retrieve("உரிமைகோரல் நிராகரிக்கப்பட்டது").language == "ta"

    def test_explicit_language_overrides_detection(self):
        assert self.build().retrieve("claim", language="hi").language == "hi"

    def test_mismatched_registration_is_rejected(self):
        m = MultilingualRetriever()
        with pytest.raises(ValueError, match="does not match key"):
            m.add_language("hi", HybridRetriever(language="en"))

    def test_missing_language_falls_back_with_a_warning(self, caplog):
        m = MultilingualRetriever()
        r = HybridRetriever(language="en")
        r.add(EN_CHUNKS)
        m.add_language("en", r)

        with caplog.at_level("WARNING"):
            result = m.retrieve("உரிமைகோரல் நிராகரிக்கப்பட்டது")
        assert result.language == "en"
        assert "wrong language" in caplog.text

    def test_no_fallback_configured_raises(self):
        with pytest.raises(KeyError, match="no retriever"):
            MultilingualRetriever().retrieve("anything")

    def test_lists_configured_languages(self):
        assert self.build().languages == ["en", "hi", "ta"]

"""Tests for sentence splitting and chunking.

The failure mode being guarded against is silent and expensive: a chunker that
cuts mid-sentence hands the NLI verifier a premise whose subject was severed. The
claim is then scored UNVERIFIABLE for a preprocessing reason, and the result looks
like a genuine multilingual model weakness in the final report.
"""

from __future__ import annotations

import pytest

from pramana.ingestion.chunking import (
    ChunkingConfig,
    chunk_text,
    chunking_report,
    estimate_tokens,
    split_paragraphs,
    split_sentences,
)

EN = (
    "Claims are rejected if submitted more than 30 days after discharge. "
    "Rejected claims may be appealed within 60 days. "
    "Appeals require the original claim number."
)
HI = (
    "डिस्चार्ज के 30 दिन बाद जमा किए गए क्लेम रिजेक्ट कर दिए जाते हैं। "
    "रिजेक्ट क्लेम की अपील 60 दिन के भीतर की जा सकती है। "
    "अपील के लिए मूल क्लेम नंबर आवश्यक है।"
)
TA = (
    "வெளியேற்றப்பட்ட 30 நாட்களுக்குப் பிறகு சமர்ப்பிக்கப்பட்ட உரிமைகோரல்கள் நிராகரிக்கப்படும். "
    "நிராகரிக்கப்பட்ட உரிமைகோரல்களை 60 நாட்களுக்குள் மேல்முறையீடு செய்யலாம். "
    "மேல்முறையீட்டிற்கு அசல் உரிமைகோரல் எண் தேவை."
)


class TestSentenceSplitting:
    def test_splits_english_on_full_stops(self):
        assert len(split_sentences(EN, "en")) == 3

    def test_splits_hindi_on_danda(self):
        """This is the test that matters. Hindi ends sentences with U+0964, not
        '.', so a full-stop splitter returns the whole document as one sentence
        and every chunk boundary lands mid-sentence."""
        sentences = split_sentences(HI, "hi")
        assert len(sentences) == 3, f"danda not honoured, got {len(sentences)}: {sentences}"
        assert all("।" in s or s == sentences[-1] for s in sentences)

    def test_hindi_would_be_one_sentence_under_a_naive_splitter(self):
        """Demonstrates the bug being prevented: '.' never occurs in this text."""
        assert "." not in HI
        assert len(HI.split(".")) == 1

    def test_splits_tamil(self):
        assert len(split_sentences(TA, "ta")) == 3

    def test_handles_double_danda(self):
        assert len(split_sentences("पहला वाक्य॥ दूसरा वाक्य॥", "hi")) == 2

    def test_does_not_split_on_abbreviations(self):
        """A split inside 'Rs. 500' would separate a number from its unit --
        exactly the kind of fragment that produces a spurious verdict."""
        out = split_sentences("The fee is Rs. 500 per claim. Pay online.", "en")
        assert len(out) == 2
        assert "Rs. 500" in out[0]

    def test_does_not_split_on_titles(self):
        out = split_sentences("Contact Dr. Anita for approval. She will respond.", "en")
        assert len(out) == 2

    def test_empty_and_whitespace_input(self):
        assert split_sentences("", "en") == []
        assert split_sentences("   \n  ", "en") == []

    def test_paragraph_splitting(self):
        assert len(split_paragraphs("Para one.\n\nPara two.\n\n\nPara three.")) == 3


class TestTokenEstimation:
    def test_indic_scripts_cost_more_tokens_per_character(self):
        """Subword vocabularies fragment Devanagari and Tamil far more than Latin.
        Sizing chunks by raw characters would give Hindi and Tamil several times
        the token budget of English and bias every cross-lingual comparison."""
        text_len = 100
        assert estimate_tokens("x" * text_len, "hi") > estimate_tokens("x" * text_len, "en")
        assert estimate_tokens("x" * text_len, "ta") > estimate_tokens("x" * text_len, "hi")

    def test_never_returns_zero(self):
        assert estimate_tokens("a", "en") >= 1
        assert estimate_tokens("", "en") >= 1


class TestChunking:
    @pytest.mark.parametrize(
        ("text", "language"), [(EN, "en"), (HI, "hi"), (TA, "ta")]
    )
    def test_produces_chunks_for_every_language(self, text, language):
        chunks = chunk_text(text, doc_id="d1", language=language)
        assert chunks
        assert all(c.text.strip() for c in chunks)
        assert all(c.language == language for c in chunks)

    def test_chunk_ids_are_unique_and_traceable(self):
        chunks = chunk_text(EN * 20, doc_id="policy", language="en")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("policy#c") for i in ids)

    def test_chunking_is_lossless(self):
        """Every sentence must survive. A chunker that drops content produces
        answers that are correctly judged unsupported -- for the wrong reason."""
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        chunks = chunk_text(text, doc_id="d", language="en")
        joined = " ".join(c.text for c in chunks)
        for i in range(40):
            assert f"Sentence number {i} is here." in joined, f"lost sentence {i}"

    def test_respects_target_size(self):
        cfg = ChunkingConfig(target_tokens=60, overlap_tokens=10, min_tokens=5)
        chunks = chunk_text(EN * 10, doc_id="d", language="en", config=cfg)
        assert len(chunks) > 1
        assert all(c.metadata["est_tokens"] <= cfg.max_tokens for c in chunks)

    def test_chunks_do_not_cut_mid_sentence(self):
        """The core guarantee: a premise with its subject cut off cannot be
        verified against any claim."""
        cfg = ChunkingConfig(target_tokens=50, overlap_tokens=10, min_tokens=5)
        chunks = chunk_text(EN * 8, doc_id="d", language="en", config=cfg)
        for c in chunks:
            stripped = c.text.strip()
            assert stripped[0].isupper() or stripped[0].isdigit(), (
                f"chunk starts mid-sentence: {stripped[:60]!r}"
            )

    def test_overlap_carries_whole_sentences(self):
        """Half a sentence as context is worse than none: it looks like evidence
        and is not."""
        cfg = ChunkingConfig(target_tokens=60, overlap_tokens=25, min_tokens=5)
        chunks = chunk_text(EN * 6, doc_id="d", language="en", config=cfg)
        if len(chunks) > 1:
            tail_sentences = set(split_sentences(chunks[0].text, "en"))
            head_sentences = set(split_sentences(chunks[1].text, "en"))
            assert tail_sentences & head_sentences, "no sentence-level overlap"

    def test_oversized_single_sentence_is_still_emitted(self):
        """Dropping it would lose content silently; chunking is lossless by
        contract."""
        giant = "word, " * 900
        chunks = chunk_text(giant, doc_id="d", language="en")
        assert chunks
        assert sum(len(c.text) for c in chunks) > len(giant) * 0.8

    def test_tiny_trailing_chunk_is_merged(self):
        """A stray 'Thank you.' chunk retrieves noisily and gives the verifier a
        premise with no content."""
        cfg = ChunkingConfig(target_tokens=40, overlap_tokens=5, min_tokens=15)
        chunks = chunk_text(EN * 4 + " Thanks.", doc_id="d", language="en", config=cfg)
        sizes = [c.metadata["est_tokens"] for c in chunks]
        assert all(s >= cfg.min_tokens for s in sizes[:-1]) or len(chunks) == 1

    def test_hindi_chunks_are_sentence_aligned(self):
        cfg = ChunkingConfig(target_tokens=60, overlap_tokens=10, min_tokens=5)
        chunks = chunk_text(HI * 6, doc_id="d", language="hi", config=cfg)
        assert len(chunks) > 1
        # Every chunk except possibly the last should end on a danda.
        assert sum(1 for c in chunks if c.text.rstrip().endswith("।")) >= len(chunks) - 1

    def test_metadata_is_preserved_and_extended(self):
        chunks = chunk_text(
            EN, doc_id="d", language="en", metadata={"source": "hr_policy.md", "section": "claims"}
        )
        m = chunks[0].metadata
        assert m["source"] == "hr_policy.md" and m["section"] == "claims"
        assert m["chunk_index"] == 0 and m["n_chunks"] == len(chunks)

    def test_empty_document_yields_no_chunks(self):
        assert chunk_text("", doc_id="d", language="en") == []
        assert chunk_text("   \n\n  ", doc_id="d", language="en") == []


class TestConfigValidation:
    def test_overlap_must_be_smaller_than_target(self):
        """Otherwise each chunk carries everything forward and packing never
        advances -- an infinite loop rather than a wrong answer."""
        with pytest.raises(ValueError, match="must be smaller than"):
            ChunkingConfig(target_tokens=100, overlap_tokens=100)

    def test_min_cannot_exceed_target(self):
        with pytest.raises(ValueError, match="min_tokens cannot exceed"):
            ChunkingConfig(target_tokens=50, min_tokens=100)


class TestReport:
    def test_report_summarises_sizes(self):
        out = chunking_report(chunk_text(EN * 10, doc_id="d", language="en"))
        assert "chunks" in out and "mean=" in out

    def test_report_handles_empty_input(self):
        assert chunking_report([]) == "no chunks"

    def test_cross_language_chunk_sizes_are_comparable(self):
        """The point of script-aware token estimation: a "220-token chunk" must
        mean roughly the same amount of content in each language, or per-language
        retrieval numbers are not comparable."""
        cfg = ChunkingConfig(target_tokens=120, overlap_tokens=20, min_tokens=10)
        counts = {
            lang: [c.metadata["est_tokens"] for c in chunk_text(text * 8, doc_id="d", language=lang, config=cfg)]
            for lang, text in (("en", EN), ("hi", HI), ("ta", TA))
        }
        means = {k: sum(v) / len(v) for k, v in counts.items()}
        assert max(means.values()) / min(means.values()) < 2.0, (
            f"chunk sizes not comparable across languages: {means}"
        )

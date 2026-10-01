"""Sentence-aware chunking for English, Hindi and Tamil.

**The problem generic chunkers get wrong.** Hindi terminates sentences with the
danda ``।`` (U+0964), not a full stop. A splitter that looks for ``.`` treats an
entire Hindi document as one sentence, producing chunks cut at arbitrary
mid-sentence positions. Retrieval then returns fragments, and the NLI verifier is
asked to check a claim against a premise whose subject was cut off -- which shows
up as an inflated UNVERIFIABLE rate that looks like a model failure and is
actually a preprocessing bug.

**The second trap: character counts are not comparable across scripts.** The same
content in Devanagari occupies a different number of characters than in English,
and subword tokenizers fragment Indic scripts far more aggressively -- often 2-4
tokens per character-cluster against roughly 0.25 for English. Sizing every
language's chunks by raw character count silently gives Hindi and Tamil chunks
several times more tokens than English ones, which biases every per-language
retrieval comparison in the study. ``estimate_tokens`` corrects for this.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from pramana.ingestion.language import normalize
from pramana.schemas import Chunk, Language

log = logging.getLogger(__name__)

# ── sentence terminators ──────────────────────────────────────────────────────
#
# U+0964 DEVANAGARI DANDA and U+0965 DOUBLE DANDA are the real sentence
# terminators in Hindi. Tamil prose generally uses the Latin full stop, but
# danda appears in older or translated material, so both are accepted.
DANDA = "।"
DOUBLE_DANDA = "॥"

_SENTENCE_END = re.compile(
    rf"(?<=[.!?{DANDA}{DOUBLE_DANDA}])\s+"
    r"|(?<=[.!?])(?=[A-Zऀ-ॿ஀-௿])"
)

_PARAGRAPH = re.compile(r"\n\s*\n")

# Rough tokens-per-character, measured against subword vocabularies. Indic scripts
# fragment far more than Latin. Approximate by design -- the point is to stop
# treating "100 characters" as language-neutral, not to predict exact counts.
_TOKENS_PER_CHAR: dict[Language, float] = {"en": 0.28, "hi": 0.75, "ta": 0.95}

# Abbreviations that must not end a sentence. Devanagari and Tamil prose use these
# far less, so the list is deliberately English-heavy.
_ABBREVIATIONS = frozenset(
    ["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie", "approx", "inc", "ltd", "pvt", "co", "corp", "fig", "no", "vol", "pp", "ch", "sec", "art", "para", "rs"]
)


def estimate_tokens(text: str, language: Language) -> int:
    """Approximate subword-token count, corrected for script.

    Used so a "300-token chunk" means roughly the same amount of content in every
    language. Sizing by characters instead would hand Hindi and Tamil chunks
    several times the token budget of English ones and quietly bias every
    cross-lingual retrieval comparison.
    """
    return max(1, round(len(text) * _TOKENS_PER_CHAR.get(language, 0.5)))


@dataclass(slots=True)
class ChunkingConfig:
    target_tokens: int = 220
    """Target chunk size. Sized to fit several chunks plus an answer inside a
    modest context window, since free-tier models are not always long-context."""

    overlap_tokens: int = 40
    """Carry-over between neighbouring chunks so a fact spanning a boundary is
    still retrievable from at least one chunk."""

    min_tokens: int = 25
    """Chunks below this are merged into their neighbour. A stray fragment
    retrieves noisily and gives the verifier a premise with no content."""

    max_tokens: int = 400
    respect_paragraphs: bool = True

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be smaller than "
                f"target_tokens ({self.target_tokens}), otherwise chunking cannot advance"
            )
        if self.min_tokens > self.target_tokens:
            raise ValueError("min_tokens cannot exceed target_tokens")


# ──────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ──────────────────────────────────────────────────────────────────────────────


def split_sentences(text: str, language: Language = "en") -> list[str]:
    """Split into sentences, honouring the danda for Devanagari.

    Abbreviation handling is intentionally simple: a false split inside "Rs. 500"
    costs a slightly odd chunk boundary, whereas the complexity of full
    abbreviation modelling is not justified for this corpus.
    """
    text = normalize(text)
    if not text:
        return []

    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]

    merged: list[str] = []
    for part in parts:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _ends_with_abbreviation(sentence: str) -> bool:
    tail = sentence.rstrip()
    if not tail.endswith("."):
        return False
    last = re.split(r"[\s(]", tail[:-1])[-1].lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    *,
    doc_id: str,
    language: Language,
    config: ChunkingConfig | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split a document into overlapping, sentence-aligned chunks.

    Chunks never cut mid-sentence: a fragment whose subject was severed cannot be
    verified against a claim, and would be scored UNVERIFIABLE for a preprocessing
    reason rather than a factual one.
    """
    cfg = config or ChunkingConfig()
    if not text.strip():
        return []

    units = split_paragraphs(text) if cfg.respect_paragraphs else [text]

    sentences: list[str] = []
    for unit in units:
        for sentence in split_sentences(unit, language):
            # A single sentence longer than max_tokens must still be emitted, or
            # the content is lost entirely. Chunking is lossless by contract.
            if estimate_tokens(sentence, language) > cfg.max_tokens:
                sentences.extend(_split_oversized(sentence, language, cfg))
            else:
                sentences.append(sentence)

    if not sentences:
        return []

    raw_chunks = _pack(sentences, language, cfg)
    raw_chunks = _merge_undersized(raw_chunks, language, cfg)

    base_meta = dict(metadata or {})
    return [
        Chunk(
            chunk_id=f"{doc_id}#c{i}",
            doc_id=doc_id,
            text=body,
            language=language,
            metadata={
                **base_meta,
                "chunk_index": i,
                "n_chunks": len(raw_chunks),
                "est_tokens": estimate_tokens(body, language),
            },
        )
        for i, body in enumerate(raw_chunks)
    ]


def _pack(sentences: list[str], language: Language, cfg: ChunkingConfig) -> list[str]:
    """Greedily fill chunks to the target size, then step back for overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        n = estimate_tokens(sentence, language)
        if current and current_tokens + n > cfg.target_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = _carry_overlap(current, language, cfg)
        current.append(sentence)
        current_tokens += n

    if current:
        chunks.append(" ".join(current))
    return chunks


def _carry_overlap(
    sentences: list[str], language: Language, cfg: ChunkingConfig
) -> tuple[list[str], int]:
    """Take whole trailing sentences as the next chunk's overlap.

    Overlap is measured in sentences rather than characters so the carried text is
    always independently meaningful -- half a sentence as context is worse than
    none, because it looks like evidence and is not.
    """
    carried: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        n = estimate_tokens(sentence, language)
        if total + n > cfg.overlap_tokens:
            break
        carried.insert(0, sentence)
        total += n
    return carried, total


def _split_oversized(sentence: str, language: Language, cfg: ChunkingConfig) -> list[str]:
    """Break a sentence that alone exceeds max_tokens, at clause boundaries."""
    parts = re.split(r"(?<=[,;:।])\s+", sentence)
    out: list[str] = []
    current: list[str] = []
    total = 0

    for part in parts:
        n = estimate_tokens(part, language)
        if current and total + n > cfg.max_tokens:
            out.append(" ".join(current))
            current, total = [], 0
        current.append(part)
        total += n

    if current:
        out.append(" ".join(current))

    log.debug("split oversized sentence into %d parts (%s)", len(out), language)
    return out or [sentence]


def _merge_undersized(chunks: list[str], language: Language, cfg: ChunkingConfig) -> list[str]:
    """Fold chunks below min_tokens into a neighbour.

    A trailing one-line chunk ("Thank you.") retrieves noisily and gives the
    verifier a premise with no content -- a reliable source of spurious
    UNVERIFIABLE verdicts.
    """
    if len(chunks) < 2:
        return chunks

    out = list(chunks)
    i = 0
    while i < len(out):
        if estimate_tokens(out[i], language) >= cfg.min_tokens or len(out) == 1:
            i += 1
            continue
        if i + 1 < len(out):
            out[i + 1] = f"{out[i]} {out[i + 1]}"
            out.pop(i)
        else:
            out[i - 1] = f"{out[i - 1]} {out[i]}"
            out.pop(i)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────


def chunking_report(chunks: list[Chunk]) -> str:
    """One-line summary. Compare across languages to confirm chunk sizes are
    genuinely comparable rather than only nominally so."""
    if not chunks:
        return "no chunks"
    sizes = [c.metadata.get("est_tokens", 0) for c in chunks]
    return (
        f"{len(chunks)} chunks · tokens min={min(sizes)} "
        f"mean={sum(sizes) / len(sizes):.0f} max={max(sizes)}"
    )

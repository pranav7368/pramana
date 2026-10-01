"""BM25 lexical retrieval with script-aware tokenisation.

**Why sparse retrieval is here at all.** Dense multilingual embeddings blur exactly
what enterprise queries depend on: policy codes, product names, claim numbers,
section references. BM25 matches those verbatim. Hybrid retrieval is not a hedge,
it covers a failure mode dense retrieval has by construction.

**Why BM25 is implemented rather than imported.** The scoring formula is thirty
lines; the part that actually matters for this project is *tokenisation*, and no
off-the-shelf implementation handles it:

* **Tamil is agglutinative.** ``சமர்ப்பிக்கப்படாததால்`` ("because it was not
  submitted") is one orthographic word carrying negation, nominalisation and
  causality. Whitespace tokenisation makes it a singleton term that matches
  nothing, so BM25 contributes zero for precisely the morphologically complex
  queries where dense retrieval is also weakest.
* **Hindi is inflectional** and uses the danda as punctuation, which a Latin
  tokeniser leaves glued to the preceding word.

Character n-grams are used for Indic scripts to recover partial matches within
long agglutinated forms.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from pramana.ingestion.language import normalize
from pramana.schemas import Chunk, Language

_LATIN_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# The Devanagari block contains its own *punctuation*: U+0964 DANDA,
# U+0965 DOUBLE DANDA, U+0970 ABBREVIATION SIGN, U+0971 HIGH SPACING DOT.
# A naive [ऀ-ॿ]+ run therefore swallows the danda into the final word of
# every Hindi sentence, so "हुआ।" and "हुआ" become different terms and the last
# word of each sentence is unreachable by query. Devanagari digits (U+0966-U+096F)
# must still be kept -- they carry the numbers this corpus turns on.
_INDIC_RUN = re.compile(
    "[ऀ-ॣ"    # Devanagari letters, vowel signs, matras
    "०-९"     # Devanagari digits -- numbers matter in this corpus
    "ॲ-ॿ"     # remaining Devanagari letters
    "஀-௿]+"   # Tamil block
)

# Deliberately short. Aggressive stop-word removal hurts this project: "not",
# "nahi" and "illai" are exactly the tokens whose loss turns a contradiction into
# an apparent agreement.
STOPWORDS: dict[Language, frozenset[str]] = {
    "en": frozenset(["a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "were", "be", "been", "by", "with", "as", "from"]),
    "hi": frozenset(["का", "की", "के", "को", "में", "से", "पर", "और", "या", "है", "हैं", "था", "थे", "यह", "वह", "इस", "उस", "एक"]),
    "ta": frozenset(["மற்றும்", "அல்லது", "ஒரு", "இந்த", "அந்த", "ஆகும்", "என்று", "உள்ள"]),
}

INDIC_NGRAM_SIZE = 4
"""Character n-gram width for Indic tokens. Four is around the length of a Tamil
morpheme -- short enough to survive agglutination, long enough to stay specific."""

MIN_INDIC_LEN_FOR_NGRAMS = 6
"""Below this, the whole token is kept as-is; n-gramming a short word only adds noise."""

# Light suffix stripping for Latin script.
#
# Added after a measured failure: the query "How long do I have to appeal?" scored
# **zero** against a corpus containing "Rejected claims may be appealed within 60
# days" -- "appeal" and "appealed" are different terms under exact matching, and the
# remaining query words are stop-words. Retrieval returned nothing and the pipeline
# correctly abstained on a question the corpus answers.
#
# Deliberately conservative: the stem is emitted *in addition to* the original
# token, never in place of it, so no information is lost and an exact match still
# outscores a stem match through term frequency. A full Porter stemmer would
# conflate more aggressively -- "policy"/"police" -- which is the wrong trade for a
# corpus where precision on domain terms is the point.
_LATIN_SUFFIXES = ("ing", "edly", "ies", "ied", "ed", "es", "ly", "s")
MIN_STEM_LEN = 4


def _stem_latin(token: str) -> str | None:
    """Strip one common English suffix. ``None`` when nothing safe can be removed."""
    if len(token) < MIN_STEM_LEN + 1 or token.isdigit():
        return None
    for suffix in _LATIN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_LEN:
            stem = token[: -len(suffix)]
            # "appealed" -> "appeal", but also normalise a doubled final consonant
            # ("submitted" -> "submitt" -> "submit").
            if len(stem) > MIN_STEM_LEN and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
    return None


def tokenize(text: str, language: Language = "en") -> list[str]:
    """Tokenise for lexical matching.

    Latin runs become words; Indic runs become the whole token *plus* its
    character n-grams, so a query sharing a stem with a longer inflected form in
    the corpus still matches.
    """
    text = unicodedata.normalize("NFC", text).lower()
    tokens: list[str] = []
    stops = STOPWORDS.get(language, frozenset())

    for match in _INDIC_RUN.finditer(text):
        token = match.group()
        if token in stops:
            continue
        tokens.append(token)
        if len(token) >= MIN_INDIC_LEN_FOR_NGRAMS:
            tokens.extend(
                token[i : i + INDIC_NGRAM_SIZE]
                for i in range(len(token) - INDIC_NGRAM_SIZE + 1)
            )

    # Digits are kept everywhere: "30 days" versus "15 days" is the single most
    # consequential distinction in this corpus.
    latin_part = _INDIC_RUN.sub(" ", text)
    for token in _LATIN_WORD.findall(latin_part):
        if token in stops:
            continue
        tokens.append(token)
        stem = _stem_latin(token)
        if stem and stem != token:
            tokens.append(stem)

    return tokens


@dataclass(slots=True)
class BM25Index:
    """Okapi BM25 over a chunk collection.

    Args:
        k1: term-frequency saturation. 1.5 is the standard default.
        b: length normalisation. 0.75 is standard; lower it if chunk lengths vary
            widely across languages, since token counts are not comparable between
            scripts even after script-aware chunking.
    """

    k1: float = 1.5
    b: float = 0.75

    chunks: list[Chunk] = field(default_factory=list)
    _doc_tokens: list[list[str]] = field(default_factory=list, repr=False)
    _doc_freqs: list[Counter[str]] = field(default_factory=list, repr=False)
    _df: Counter[str] = field(default_factory=Counter, repr=False)
    _doc_len: list[int] = field(default_factory=list, repr=False)
    _avg_len: float = 0.0

    # ── build ─────────────────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(normalize(chunk.text), chunk.language)
            self.chunks.append(chunk)
            self._doc_tokens.append(tokens)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._df.update(freqs.keys())
            self._doc_len.append(len(tokens))
        self._avg_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0

    @property
    def size(self) -> int:
        return len(self.chunks)

    # ── query ─────────────────────────────────────────────────────────────────

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF, floored at zero.

        The raw formula goes negative for terms appearing in more than half the
        collection. On a small corpus that is common, and a negative contribution
        would let a document be *penalised* for containing a query term.
        """
        n = self.size
        df = self._df.get(term, 0)
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(
        self, query: str, top_k: int = 10, language: Language | None = None
    ) -> list[tuple[int, float]]:
        """Return ``(chunk_index, score)`` pairs, best first."""
        if not self.size:
            return []

        lang = language or (self.chunks[0].language if self.chunks else "en")
        terms = tokenize(normalize(query), lang)
        if not terms:
            return []

        scores = [0.0] * self.size
        for term in set(terms):
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for i, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * (self._doc_len[i] / self._avg_len or 1.0)
                scores[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0.0), key=lambda x: -x[1]
        )
        return ranked[:top_k]

    def explain(self, query: str, chunk_index: int, language: Language | None = None) -> dict[str, float]:
        """Per-term score contributions for one chunk.

        Retrieval failures are otherwise very hard to diagnose: this shows whether
        a Tamil query missed because its n-grams did not match, or because the
        term is too common to carry signal.
        """
        lang = language or self.chunks[chunk_index].language
        freqs = self._doc_freqs[chunk_index]
        out: dict[str, float] = {}
        for term in set(tokenize(normalize(query), lang)):
            tf = freqs.get(term, 0)
            if not tf:
                continue
            idf = self._idf(term)
            norm = 1 - self.b + self.b * (self._doc_len[chunk_index] / self._avg_len or 1.0)
            out[term] = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        return dict(sorted(out.items(), key=lambda x: -x[1]))

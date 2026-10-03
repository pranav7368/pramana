"""Bounded, in-memory document preparation for the localhost demonstration.

This module never writes uploads to disk. Pilot ingestion remains the separately
approved Markdown/text process described in the pilot runbook.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from pramana.ingestion.chunking import chunk_text
from pramana.schemas import Chunk, Language

MAX_UPLOAD_BYTES = 5_000_000
MAX_PDF_PAGES = 25
MAX_EXTRACTED_CHARS = 120_000
MAX_DEMO_CHUNKS = 300


class DocumentError(ValueError):
    """An upload cannot safely or usefully be indexed for the demo."""


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    name: str
    language: Language
    sha256: str
    pages: int
    characters: int
    chunks: list[Chunk]

    def summary(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "language": self.language,
            "sha256": self.sha256,
            "pages": self.pages,
            "characters": self.characters,
            "chunks": len(self.chunks),
        }


def _safe_name(filename: str) -> str:
    # A user-supplied name is display metadata, never a path to open or save.
    name = filename.replace("\\", "/").split("/")[-1].strip()
    if not name or len(name) > 120 or not re.fullmatch(r"[^\x00-\x1f\x7f]+", name):
        raise DocumentError("Choose a file with a short, valid name.")
    return name


def prepare_document(data: bytes, filename: str, language: Language | None = None) -> PreparedDocument:
    """Extract, chunk and fingerprint an upload; ``language=None`` detects it from the text."""
    name = _safe_name(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise DocumentError("Only text PDFs, .md, and .txt files are supported.")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError("File must be non-empty and at most 5 MB.")

    if suffix == ".pdf":
        pages = _extract_pdf_pages(data)
    else:
        if b"\x00" in data:
            raise DocumentError("The text file contains binary data.")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError("Text files must use UTF-8 encoding.") from exc
        pages = [(1, text)]

    if sum(len(text) for _, text in pages) > MAX_EXTRACTED_CHARS:
        raise DocumentError("Extracted text exceeds the 120,000-character demo limit.")
    if language is None:
        from pramana.ingestion.language import detect_language
        language = detect_language(" ".join(text for _, text in pages)[:4000]).language
    digest = hashlib.sha256(data).hexdigest()
    chunks: list[Chunk] = []
    for page_number, text in pages:
        chunks.extend(chunk_text(
            text, doc_id=f"upload_{digest[:12]}_p{page_number}", language=language,
            metadata={"source": name, "page": page_number, "sha256": digest},
        ))
        if len(chunks) > MAX_DEMO_CHUNKS:
            raise DocumentError("Document produces too many searchable passages for the demo.")
    if not chunks:
        raise DocumentError("No selectable text found. Run OCR on scanned pages, then upload again.")
    return PreparedDocument(name, language, digest, len(pages), sum(len(t) for _, t in pages), chunks)


def _extract_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    if not data.startswith(b"%PDF-"):
        raise DocumentError("This file is not a valid PDF.")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentError("PDF support is missing. Run the one-command demo setup again.") from exc
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise DocumentError("Password-protected PDFs are not supported in the demo.")
        if not 1 <= len(reader.pages) <= MAX_PDF_PAGES:
            raise DocumentError("PDFs must contain 1–25 pages.")
        pages: list[tuple[int, str]] = []
        characters = 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            characters += len(text)
            if characters > MAX_EXTRACTED_CHARS:
                raise DocumentError("Extracted text exceeds the 120,000-character demo limit.")
            if text.strip():
                pages.append((number, text))
        if not pages:
            raise DocumentError("No selectable text found. Run OCR on scanned pages, then upload again.")
        return pages
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("PDF text could not be read. Try a text-based PDF or UTF-8 text file.") from exc

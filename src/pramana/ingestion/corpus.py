"""Load approved UTF-8 Markdown/text corpora while preserving document provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pramana.ingestion.chunking import chunk_text
from pramana.schemas import Chunk, Language


def load_corpus(
    root: Path, languages: tuple[Language, ...] = ("en", "hi", "ta"), *,
    max_file_bytes: int = 2_000_000, max_documents: int = 500, max_chunks: int = 10000,
) -> dict[Language, list[Chunk]]:
    root = root.resolve(strict=True)
    result: dict[Language, list[Chunk]] = {}
    documents = total_chunks = 0
    for language in languages:
        folder = root / language
        chunks: list[Chunk] = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            # Corpus administrators choose files, but links must not import outside data.
            if not path.resolve().is_relative_to(root):
                raise ValueError("Corpus links must stay inside the corpus directory")
            documents += 1
            if documents > max_documents or path.stat().st_size > max_file_bytes:
                raise ValueError("Corpus exceeds its document count or file size limit")
            raw = path.read_bytes()
            if len(raw) > max_file_bytes:
                raise ValueError("Corpus file exceeds size limit")
            text = raw.decode("utf-8-sig")
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256(raw).hexdigest()
            loaded = chunk_text(
                text, doc_id=f"{relative}@{digest[:12]}", language=language,
                metadata={"source": relative, "sha256": digest},
            )
            total_chunks += len(loaded)
            if total_chunks > max_chunks:
                raise ValueError("Corpus exceeds chunk limit")
            chunks.extend(loaded)
        if not chunks:
            raise ValueError(f"No non-empty Markdown/text documents for language {language}")
        result[language] = chunks
    return result

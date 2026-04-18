"""
KG-MAG — Data Ingestion Pipeline
=================================
Handles: PDF, Markdown (.md), plain text (.txt)

Stages
------
1. File type detection & routing
2. Raw text extraction
3. Text cleaning & normalization
4. Metadata extraction
5. Semantic chunking (heading-aware, overlapping)
6. Emit DocumentChunk objects ready for embedding

Design notes
------------
- Heading-aware chunking: we detect markdown/PDF headings and use them
  as natural breakpoints, then apply sliding-window overlap inside sections.
- Overlap prevents retrieval gaps at chunk boundaries.
- Each chunk carries provenance metadata for traceable citations.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import structlog

from backend.core.config import get_settings
from backend.core.models import DocumentChunk, DocumentMetadata, SourceType

logger = structlog.get_logger(__name__)


# ── Text Cleaning ─────────────────────────────────────────────────────────────


def clean_text(raw: str) -> str:
    """
    Normalize and clean raw text extracted from any source.
    - Normalize unicode to NFC
    - Remove null bytes, soft hyphens, zero-width chars
    - Collapse excessive whitespace / blank lines
    - Preserve paragraph structure
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", raw)

    # Strip control characters (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f\xad\u200b-\u200f\ufeff]", "", text)

    # Fix ligatures / common OCR artifacts
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")

    # Collapse repeated whitespace on a single line (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive newlines → double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── PDF Extraction ────────────────────────────────────────────────────────────


def extract_pdf(file_path: Path) -> tuple[str, DocumentMetadata]:
    """
    Extract text and metadata from a PDF file using pdfplumber (primary)
    with pymupdf as fallback for scanned/complex PDFs.
    Returns (full_text, metadata).
    """
    try:
        import pdfplumber  # type: ignore

        pages: list[str] = []
        title: str | None = None
        author: str | None = None

        with pdfplumber.open(file_path) as pdf:
            info = pdf.metadata or {}
            title = info.get("Title") or None
            author = info.get("Author") or None
            page_count = len(pdf.pages)

            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

                # Some PDFs return empty text blocks but still expose word boxes.
                if not page_text.strip():
                    try:
                        words = page.extract_words(x_tolerance=2, y_tolerance=2) or []
                        page_text = " ".join(w.get("text", "") for w in words).strip()
                    except Exception:
                        page_text = ""

                pages.append(page_text)

        full_text = "\n\n".join(p for p in pages if p.strip())

        if not full_text.strip():
            # Fallback: pdfplumber extracted nothing → try pymupdf
            logger.info(
                "Pdfplumber extracted no text, trying pymupdf fallback",
                path=str(file_path),
            )
            full_text = _extract_pdf_pymupdf(file_path)

    except ImportError:
        logger.info("pdfplumber not available, using pymupdf", path=str(file_path))
        full_text = _extract_pdf_pymupdf(file_path)
        page_count = 0
        title = None
        author = None
    except Exception as e:
        logger.error(
            "PDF extraction failed",
            file=str(file_path),
            error=str(e),
            error_type=type(e).__name__,
        )
        raise

    cleaned = clean_text(full_text)
    meta = DocumentMetadata(
        filename=file_path.name,
        source_type=SourceType.PDF,
        page_count=page_count,
        word_count=len(cleaned.split()),
        title=title or file_path.stem,
        author=author,
    )
    return cleaned, meta


def _extract_pdf_pymupdf(file_path: Path) -> str:
    try:
        import fitz  # type: ignore  # PyMuPDF

        doc = fitz.open(str(file_path))
        return "\n\n".join(page.get_text() for page in doc)
    except ImportError as e:
        raise RuntimeError(
            "Neither pdfplumber nor pymupdf (fitz) are installed. "
            "Run: pip install pdfplumber pymupdf"
        ) from e
    except Exception as e:
        logger.error(
            "PyMuPDF extraction failed",
            file=str(file_path),
            error=str(e),
            error_type=type(e).__name__,
        )
        raise RuntimeError(f"Failed to extract PDF with pymupdf: {str(e)}") from e


# ── Markdown Extraction ───────────────────────────────────────────────────────


def extract_markdown(file_path: Path) -> tuple[str, DocumentMetadata]:
    """
    Parse a markdown file.  We keep the heading structure intact
    (it will be used as chunk boundary hints later).
    """
    raw = file_path.read_text(encoding="utf-8", errors="replace")

    # Extract YAML front-matter if present
    title: str | None = None
    author: str | None = None
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if fm_match:
        fm_block = fm_match.group(1)
        if m := re.search(r"^title:\s*(.+)$", fm_block, re.MULTILINE):
            title = m.group(1).strip().strip('"').strip("'")
        if m := re.search(r"^author:\s*(.+)$", fm_block, re.MULTILINE):
            author = m.group(1).strip()
        raw = raw[fm_match.end() :]

    cleaned = clean_text(raw)
    meta = DocumentMetadata(
        filename=file_path.name,
        source_type=SourceType.MARKDOWN,
        word_count=len(cleaned.split()),
        title=title or file_path.stem,
        author=author,
    )
    return cleaned, meta


# ── Plain Text Extraction ─────────────────────────────────────────────────────


def extract_text(file_path: Path) -> tuple[str, DocumentMetadata]:
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    meta = DocumentMetadata(
        filename=file_path.name,
        source_type=SourceType.TEXT,
        word_count=len(cleaned.split()),
        title=file_path.stem,
    )
    return cleaned, meta


# ── Semantic + Contextual Chunker ───────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+")


def _normalize_heading(line: str) -> str:
    return line.strip().lstrip("#").strip().rstrip(":").strip()


def _is_heading_candidate(paragraph: str) -> bool:
    """Heuristic heading detection across markdown, outlines, and PDF-extracted text."""
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    if not lines:
        return False

    first = lines[0]
    words = first.split()

    # Markdown headings.
    if re.match(r"^#{1,6}\s+\S+", first):
        return True

    # Numbered headings (e.g., "2.1 Design Decisions").
    if re.match(r"^\d+(?:\.\d+){0,3}\s+\S+", first) and len(words) <= 16:
        return True

    # ALL CAPS short lines often indicate PDF headings.
    alpha_count = sum(ch.isalpha() for ch in first)
    if alpha_count >= 4 and first == first.upper() and len(words) <= 14:
        return True

    # Short colon-ended lines are often heading labels.
    if first.endswith(":") and len(words) <= 14:
        return True

    return False


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text.strip() else []


def _paragraph_to_semantic_units(
    paragraph: str,
    heading: str | None,
    max_unit_words: int,
) -> list[tuple[str | None, str]]:
    """Split long paragraphs into sentence-grouped semantic units."""
    units: list[tuple[str | None, str]] = []
    sentences = _split_sentences(paragraph)
    if not sentences:
        return units

    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        if current and (current_words + len(sentence_words) > max_unit_words):
            units.append((heading, " ".join(current).strip()))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += len(sentence_words)

    if current:
        units.append((heading, " ".join(current).strip()))

    return units


def _semantic_units(text: str, chunk_size: int) -> list[tuple[str | None, str]]:
    """
    Convert raw text into semantic units while preserving heading context.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    max_unit_words = max(80, chunk_size // 2)

    units: list[tuple[str | None, str]] = []
    current_heading: str | None = None

    for paragraph in paragraphs:
        if _is_heading_candidate(paragraph):
            lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
            heading = _normalize_heading(lines[0])
            current_heading = heading or current_heading

            # If this block also contains body text, keep it with heading context.
            trailing = "\n".join(lines[1:]).strip()
            if trailing:
                units.extend(
                    _paragraph_to_semantic_units(
                        trailing, current_heading, max_unit_words
                    )
                )
            continue

        units.extend(
            _paragraph_to_semantic_units(paragraph, current_heading, max_unit_words)
        )

    if not units and text.strip():
        units.extend(_paragraph_to_semantic_units(text.strip(), None, max_unit_words))

    return units


def _semantic_contextual_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str | None, str]]:
    """
    Build contextual chunks from semantic units.
    - semantic: paragraph/sentence-aware grouping
    - contextual: heading label injection + overlap windows
    """
    units = _semantic_units(text, chunk_size)
    if not units:
        return []

    chunks: list[tuple[str | None, str]] = []
    buffer_words: list[str] = []
    words_since_emit = 0
    current_heading: str | None = None
    step_overlap = max(0, min(overlap, chunk_size - 1))

    def emit_chunk(heading: str | None) -> None:
        nonlocal buffer_words
        chunk_text = " ".join(buffer_words).strip()
        if not chunk_text:
            return
        chunks.append((heading, chunk_text))
        if step_overlap > 0:
            buffer_words = buffer_words[-step_overlap:]
        else:
            buffer_words = []

    for heading, unit_text in units:
        if heading:
            current_heading = heading

        # Inject heading into unit text to preserve local context for retrieval.
        contextual_text = unit_text
        if heading:
            contextual_text = f"{heading}. {unit_text}" if unit_text else heading

        words = contextual_text.split()
        if not words:
            continue

        cursor = 0
        while cursor < len(words):
            remaining = chunk_size - len(buffer_words)
            if remaining <= 0:
                emit_chunk(current_heading)
                words_since_emit = 0
                remaining = chunk_size - len(buffer_words)

            take = words[cursor : cursor + remaining]
            if not take:
                break
            buffer_words.extend(take)
            words_since_emit += len(take)
            cursor += len(take)

            if len(buffer_words) >= chunk_size:
                emit_chunk(current_heading)
                words_since_emit = 0

    # Flush only if new words were added since the last emit.
    if buffer_words and words_since_emit > 0:
        min_tail_words = max(20, chunk_size // 8)
        tail_text = " ".join(buffer_words).strip()

        # Avoid creating tiny, low-value tail chunks when we already have chunks.
        if chunks and len(buffer_words) < min_tail_words:
            prev_heading, prev_text = chunks[-1]
            chunks[-1] = (prev_heading, f"{prev_text} {tail_text}".strip())
        else:
            chunks.append((current_heading, tail_text))

    return chunks


def _sliding_window_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split a block of text into overlapping chunks by word count.
    This is the inner chunker applied within each section.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - max(0, overlap))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def chunk_document(
    text: str,
    meta: DocumentMetadata,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Main chunking entry point.

    Strategy
    --------
    1. Split on headings → preserve section context
    2. Within each section, apply sliding-window overlap
    3. Each chunk carries heading, source metadata, stable chunk_id
    """
    cfg = get_settings()
    chunk_size = chunk_size or cfg.chunk_size
    overlap = overlap or cfg.chunk_overlap

    contextual_chunks = _semantic_contextual_chunks(text, chunk_size, overlap)

    # Fallback path for malformed text where semantic parsing yields no chunks.
    if not contextual_chunks and text.strip():
        fallback_chunks = _sliding_window_chunks(text, chunk_size, overlap)
        contextual_chunks = [(None, c) for c in fallback_chunks]

    chunks: list[DocumentChunk] = []
    idx = 0

    for heading, chunk_text in contextual_chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # Deterministic chunk_id based on stable document metadata + chunk content.
        content_hash = hashlib.sha256(
            (
                f"{meta.filename}:{meta.title or ''}:{meta.word_count or 0}:"
                f"{idx}:{heading or ''}:{chunk_text}"
            ).encode("utf-8")
        ).hexdigest()[:16]

        chunks.append(
            DocumentChunk(
                chunk_id=content_hash,
                source_id=meta.source_id,
                filename=meta.filename,
                text=chunk_text,
                chunk_index=idx,
                heading=heading,
                token_count=len(chunk_text.split()),
                ingested_at=meta.ingested_at,
            )
        )
        idx += 1

    if not chunks and text.strip():
        # Last-resort guardrail: always emit at least one chunk for non-empty text.
        content_hash = hashlib.sha256(
            (
                f"{meta.filename}:{meta.title or ''}:{meta.word_count or 0}:"
                f"0::{text.strip()}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        chunks.append(
            DocumentChunk(
                chunk_id=content_hash,
                source_id=meta.source_id,
                filename=meta.filename,
                text=text.strip(),
                chunk_index=0,
                heading=None,
                token_count=len(text.split()),
                ingested_at=meta.ingested_at,
            )
        )

    logger.info(
        "Chunked document",
        filename=meta.filename,
        sections=len(_semantic_units(text, chunk_size)),
        chunks=len(chunks),
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunks


# ── Dispatcher ────────────────────────────────────────────────────────────────

EXTRACTORS = {
    ".pdf": extract_pdf,
    ".md": extract_markdown,
    ".markdown": extract_markdown,
    ".txt": extract_text,
    ".text": extract_text,
}


def ingest_file(file_path: Path) -> list[DocumentChunk]:
    """
    Full ingestion pipeline for a single file.
    Returns a list of DocumentChunk objects ready for embedding.
    """
    suffix = file_path.suffix.lower()
    extractor = EXTRACTORS.get(suffix)

    if extractor is None:
        supported = ", ".join(EXTRACTORS.keys())
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    logger.info("Ingesting file", path=str(file_path), type=suffix)
    text, meta = extractor(file_path)

    if not text.strip():
        logger.warning("Empty document — skipping", filename=file_path.name)
        return []

    return chunk_document(text, meta)


def ingest_directory(directory: Path) -> Iterator[list[DocumentChunk]]:
    """
    Walk a directory and ingest all supported files.
    Yields chunks per file (lazy — memory efficient for large corpora).
    """
    supported_exts = set(EXTRACTORS.keys())
    files = [
        f
        for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_exts
    ]

    logger.info("Starting directory ingestion", path=str(directory), files=len(files))

    for fp in sorted(files):
        try:
            chunks = ingest_file(fp)
            yield chunks
        except Exception as e:
            logger.error("Failed to ingest file", path=str(fp), error=str(e))
            continue

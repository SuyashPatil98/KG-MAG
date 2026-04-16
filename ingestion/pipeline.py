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
import io
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
                pages.append(page_text)

        full_text = "\n\n".join(p for p in pages if p.strip())

        if not full_text.strip():
            # Fallback: pdfplumber extracted nothing → try pymupdf
            logger.info("Pdfplumber extracted no text, trying pymupdf fallback", path=str(file_path))
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
            error_type=type(e).__name__
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
            error_type=type(e).__name__
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
        raw = raw[fm_match.end():]

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


# ── Heading-Aware Semantic Chunker ────────────────────────────────────────────

# Matches markdown headings (# to ####) or ALL-CAPS lines ≥4 chars (PDF headings)
_HEADING_RE = re.compile(
    r"^(#{1,4}\s+.+|[A-Z][A-Z\s\d]{3,}(?:\n|$))", re.MULTILINE
)


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split text into (heading, body) pairs.
    Returns a list of sections preserving document order.
    """
    sections: list[tuple[str | None, str]] = []
    last_end = 0
    current_heading: str | None = None

    for match in _HEADING_RE.finditer(text):
        body = text[last_end:match.start()].strip()
        if body:
            sections.append((current_heading, body))
        current_heading = match.group(0).strip().lstrip("#").strip()
        last_end = match.end()

    # Remaining text after last heading
    tail = text[last_end:].strip()
    if tail:
        sections.append((current_heading, tail))

    return sections


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

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

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

    sections = _split_into_sections(text)
    chunks: list[DocumentChunk] = []
    idx = 0

    for heading, body in sections:
        if not body.strip():
            continue

        sub_chunks = _sliding_window_chunks(body, chunk_size, overlap)
        for sc in sub_chunks:
            sc = sc.strip()
            if not sc:
                continue

            # Deterministic chunk_id based on content hash
            content_hash = hashlib.sha256(
                f"{meta.source_id}:{idx}:{sc[:64]}".encode()
            ).hexdigest()[:16]

            chunks.append(
                DocumentChunk(
                    chunk_id=content_hash,
                    source_id=meta.source_id,
                    filename=meta.filename,
                    text=sc,
                    chunk_index=idx,
                    heading=heading,
                    token_count=len(sc.split()),
                    ingested_at=meta.ingested_at,
                )
            )
            idx += 1

    logger.info(
        "Chunked document",
        filename=meta.filename,
        sections=len(sections),
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
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {supported}"
        )

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
        f for f in directory.rglob("*")
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

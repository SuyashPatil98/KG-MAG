"""
KG-MAG — Core Pydantic data models.
These schemas flow through the entire pipeline and across API boundaries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Document / Chunk Models ───────────────────────────────────────────────────

class SourceType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    source_type: SourceType
    page_count: int | None = None
    word_count: int | None = None
    title: str | None = None
    author: str | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    filename: str
    text: str
    chunk_index: int
    heading: str | None = None          # Section heading if detected
    page_number: int | None = None
    token_count: int | None = None
    embedding: list[float] | None = None   # Populated during indexing
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store with its relevance score."""
    chunk: DocumentChunk
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score")
    rank: int = Field(ge=1)


# ── Article Models ────────────────────────────────────────────────────────────

class ArticleSection(BaseModel):
    heading: str
    content: str
    citations: list[str] = Field(default_factory=list)   # chunk_ids cited
    image_url: str | None = None
    image_prompt: str | None = None


class ArticleOutline(BaseModel):
    title: str
    subtitle: str
    target_audience: str
    estimated_reading_time: int          # minutes
    sections: list[str]                  # section headings
    seo_keywords: list[str] = Field(default_factory=list)


class GeneratedArticle(BaseModel):
    article_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    title: str
    subtitle: str
    header_image_url: str | None = None
    sections: list[ArticleSection]
    conclusion: str
    citations_map: dict[str, DocumentChunk] = Field(default_factory=dict)
    seo_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)


# ── QA Models ────────────────────────────────────────────────────────────────

class GroundingResult(BaseModel):
    sentence: str
    is_grounded: bool
    supporting_chunk_ids: list[str]
    confidence: float


class QAReport(BaseModel):
    article_id: str
    grounding_score: float          # 0–1 fraction of grounded sentences
    readability_score: float        # Flesch Reading Ease 0–100
    coverage_score: float           # % retrieved chunks actually cited
    consistency_score: float        # Self-consistency (0–1)
    overall_confidence: float       # Weighted composite
    grounding_details: list[GroundingResult]
    warnings: list[str] = Field(default_factory=list)
    passed: bool


# ── API Request / Response Models ────────────────────────────────────────────

class IngestRequest(BaseModel):
    rebuild_index: bool = False


class IngestResponse(BaseModel):
    job_id: str
    status: str
    chunks_created: int
    documents_processed: int
    duration_seconds: float


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500)
    target_audience: str = Field(default="general tech readers")
    tone: str = Field(default="informative and engaging")
    generate_images: bool = Field(default=True)
    run_qa: bool = Field(default=True)
    max_sections: int = Field(default=6, ge=2, le=10)


class GenerateResponse(BaseModel):
    article_id: str
    status: str
    article: GeneratedArticle | None = None
    qa_report: QAReport | None = None
    duration_seconds: float


class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    total_chunks: int
    index_built: bool
    vector_db: str
    embedding_model: str
    last_updated: datetime | None

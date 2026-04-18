"""
KG-MAG — Central configuration module.
Loads and validates all settings from environment variables.
Uses pydantic-settings for type-safe, validated config.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single source of truth for all KG-MAG configuration.
    Values are loaded from environment variables or the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (optional in dev/tests)",
    )
    llm_model: str = Field(default="gpt-4.1", description="OpenAI model identifier")
    max_article_tokens: int = Field(default=4096, ge=512, le=8192)
    planner_max_tokens: int = Field(default=320, ge=96, le=1024)
    writer_source_word_budget: int = Field(default=900, ge=200, le=3000)
    writer_max_sources_per_section: int = Field(default=3, ge=1, le=10)
    writer_section_max_tokens: int = Field(default=560, ge=128, le=2048)
    writer_conclusion_max_tokens: int = Field(default=280, ge=64, le=1024)

    # ── Image Generation ─────────────────────────────────────────────────────
    nanobananpro_api_key: str = Field(
        default="",
        description="Nanobananpro image API key (optional in dev/tests)",
    )
    nanobananpro_api_url: AnyHttpUrl = Field(
        default="https://api.nanobananpro.com/v1/generate"
    )

    # ── Vector DB ────────────────────────────────────────────────────────────
    vector_db: Literal["faiss", "chroma"] = Field(default="faiss")
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    embedding_batch_size: int = Field(default=64, ge=1)

    # ── Storage paths ────────────────────────────────────────────────────────
    knowledge_base_path: Path = Field(default=Path("./data/kb"))
    uploads_path: Path = Field(default=Path("./data/uploads"))
    artifacts_path: Path = Field(default=Path("./data/artifacts"))
    faiss_index_path: Path = Field(default=Path("./data/kb/faiss_index"))

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(default=8, ge=1, le=50)
    rerank_top_k: int = Field(default=4, ge=1, le=20)

    # ── QA thresholds ────────────────────────────────────────────────────────
    qa_grounding_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    qa_readability_min: float = Field(default=50.0, ge=0.0, le=100.0)
    qa_grounding_mode: Literal["heuristic", "llm"] = Field(default="heuristic")
    qa_paragraph_checks_per_section: int = Field(default=2, ge=1, le=5)

    # ── API server ───────────────────────────────────────────────────────────
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    backend_api_key: str = Field(
        default="", description="Bearer auth key for API (required in production)"
    )

    # ── Security controls ────────────────────────────────────────────────────
    max_upload_files_per_request: int = Field(default=10, ge=1, le=100)
    max_upload_file_size_mb: int = Field(default=20, ge=1, le=500)
    rate_limit_window_seconds: int = Field(default=60, ge=10, le=3600)
    rate_limit_generate_requests: int = Field(default=6, ge=1, le=200)
    rate_limit_ingest_requests: int = Field(default=8, ge=1, le=200)
    rate_limit_management_requests: int = Field(default=30, ge=1, le=500)

    # ── Runtime ──────────────────────────────────────────────────────────────
    environment: Literal["development", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    def ensure_paths(self) -> None:
        """Create all required data directories if they don't exist."""
        for p in [
            self.knowledge_base_path,
            self.uploads_path,
            self.artifacts_path,
            self.faiss_index_path.parent,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    Import this function rather than instantiating Settings directly.
    """
    s = Settings()  # type: ignore[call-arg]
    s.ensure_paths()
    return s

"""
KG-MAG — FastAPI Backend Application
======================================
Production-grade REST API for the Knowledge-Grounded Medium Article Generator.

Endpoints
---------
POST /api/ingest          — Upload and index documents
GET  /api/kb/status       — Knowledge base statistics
POST /api/generate        — Generate a grounded article
GET  /api/article/{id}    — Retrieve a generated article
GET  /api/artifacts/{file} — Serve generated images
GET  /health              — Health check
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import structlog
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from backend.agents.orchestrator import ArticleOrchestrator
from backend.core.config import get_settings
from backend.core.logging import get_logger, setup_logging
from backend.core.models import (
    GenerateRequest,
    GenerateResponse,
    IngestResponse,
    KnowledgeBaseStatus,
)
from backend.tools.llm_client import LLMClient
from backend.tools.vector_store import FAISSVectorStore
from ingestion.pipeline import ingest_file

logger = get_logger(__name__)

# ── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()
    setup_logging(cfg.log_level, cfg.environment)

    app = FastAPI(
        title="KG-MAG API",
        description="Knowledge-Grounded Medium Article Generator",
        version="1.0.0",
        docs_url="/docs" if cfg.environment == "development" else None,
        redoc_url="/redoc" if cfg.environment == "development" else None,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Dependency injection ──────────────────────────────────────────────────
    _vector_store = FAISSVectorStore()
    _llm = LLMClient()
    _orchestrator = ArticleOrchestrator(_vector_store, _llm)

    # In-memory job store (use Redis in production)
    _jobs: dict[str, dict] = {}
    _articles: dict[str, dict] = {}

    # ── Security ──────────────────────────────────────────────────────────────
    security = HTTPBearer(auto_error=False)

    def verify_api_key(
        credentials: HTTPAuthorizationCredentials | None = Security(security),
    ) -> None:
        api_key = cfg.backend_api_key
        if not api_key:
            return   # Auth disabled
        if not credentials or credentials.credentials != api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "environment": cfg.environment,
            "vector_db": cfg.vector_db,
            "total_vectors": _vector_store.total_vectors,
        }

    @app.get("/api/kb/status", response_model=KnowledgeBaseStatus)
    async def kb_status(_: None = Depends(verify_api_key)):
        chunks = _vector_store.all_chunks()
        sources = {c.source_id for c in chunks}
        # Get the most recent ingestion timestamp
        last_updated = max((c.ingested_at for c in chunks), default=None)
        return KnowledgeBaseStatus(
            total_documents=len(sources),
            total_chunks=len(chunks),
            index_built=_vector_store.total_vectors > 0,
            vector_db=cfg.vector_db,
            embedding_model=cfg.embedding_model,
            last_updated=last_updated,
        )

    @app.post("/api/ingest", response_model=IngestResponse)
    async def ingest_documents(
        files: list[UploadFile] = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        _: None = Depends(verify_api_key),
    ):
        job_id = str(uuid.uuid4())
        logger.info("Ingest endpoint hit", job_id=job_id, files_count=len(files))
        t0 = time.perf_counter()

        allowed_exts = {".pdf", ".md", ".markdown", ".txt", ".text"}
        saved_paths: list[Path] = []

        try:
            for upload in files:
                suffix = Path(upload.filename or "file.txt").suffix.lower()
                if suffix not in allowed_exts:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_exts}",
                    )
                dest = cfg.uploads_path / f"{uuid.uuid4().hex}{suffix}"
                dest.write_bytes(await upload.read())
                saved_paths.append(dest)

            # Process synchronously for simplicity; move to Celery/ARQ in production
            def _process_file(file_path):
                c = ingest_file(file_path)
                return _vector_store.add_chunks(c)

            total_chunks = 0
            for fp in saved_paths:
                try:
                    added = await run_in_threadpool(_process_file, fp)
                    total_chunks += added
                except Exception as e:
                    logger.error(
                        "Failed to ingest file",
                        file=str(fp),
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Continue with next file instead of crashing
                    continue

            await run_in_threadpool(_vector_store.save)

            # Train/update reranker in background
            background_tasks.add_task(_retrain_reranker, _vector_store)

            elapsed = time.perf_counter() - t0
            return IngestResponse(
                job_id=job_id,
                status="completed",
                chunks_created=total_chunks,
                documents_processed=len(saved_paths),
                duration_seconds=round(elapsed, 2),
            )
        except HTTPException:
            # Re-raise HTTP exceptions (file type validation, etc)
            raise
        except Exception as e:
            logger.error(
                "Ingestion pipeline failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {str(e)}"
            )

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate_article(
        request: GenerateRequest,
        _: None = Depends(verify_api_key),
    ):
        if _vector_store.total_vectors == 0:
            raise HTTPException(
                status_code=422,
                detail="Knowledge base is empty. Please ingest documents first via POST /api/ingest",
            )

        t0 = time.perf_counter()

        try:
            # Add timeout to prevent indefinite hangs (Gemini API timeouts, etc)
            try:
                ctx = await asyncio.wait_for(
                    _orchestrator.run(
                        topic=request.topic,
                        target_audience=request.target_audience,
                        tone=request.tone,
                        generate_images=request.generate_images,
                        run_qa=request.run_qa,
                        max_sections=request.max_sections,
                    ),
                    timeout=300.0  # 5 minute timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Generation pipeline timed out",
                    timeout_seconds=300,
                )
                raise HTTPException(
                    status_code=504,
                    detail="Generation request timed out after 5 minutes"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Generation pipeline failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

        if ctx.article:
            _articles[ctx.article.article_id] = ctx.article.model_dump()

        elapsed = time.perf_counter() - t0
        return GenerateResponse(
            article_id=ctx.article.article_id if ctx.article else "",
            status="completed" if ctx.article else "failed",
            article=ctx.article,
            qa_report=ctx.qa_report,
            duration_seconds=round(elapsed, 1),
        )

    @app.get("/api/article/{article_id}")
    async def get_article(article_id: str, _: None = Depends(verify_api_key)):
        article = _articles.get(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article

    @app.get("/api/articles")
    async def list_articles(_: None = Depends(verify_api_key)):
        return [
            {"article_id": k, "title": v.get("title"), "topic": v.get("topic")}
            for k, v in _articles.items()
        ]

    @app.get("/artifacts/{filename}")
    async def serve_artifact(filename: str):
        """Serve generated images."""
        fp = cfg.artifacts_path / filename
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(fp)

    @app.delete("/api/kb/clear")
    async def clear_knowledge_base(_: None = Depends(verify_api_key)):
        _vector_store.clear()
        return {"status": "cleared"}

    return app


def _retrain_reranker(vector_store: FAISSVectorStore) -> None:
    """Background task: retrain reranker after new documents are ingested."""
    chunks = vector_store.all_chunks()
    if len(chunks) < 10:
        return
    try:
        from backend.models.reranker import train_reranker
        train_reranker(chunks)
    except Exception as e:
        logger.error("Reranker training failed", error=str(e))


# ── Entry Point ───────────────────────────────────────────────────────────────

app = create_app()


if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host=cfg.backend_host,
        port=cfg.backend_port,
        reload=cfg.environment == "development",
        log_level=cfg.log_level.lower(),
    )

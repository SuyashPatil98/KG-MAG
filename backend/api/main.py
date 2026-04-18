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
import re
import secrets
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from backend.agents.orchestrator import ArticleOrchestrator
from backend.core.config import get_settings
from backend.core.logging import get_logger, setup_logging
from backend.core.models import (
    DashboardMetrics,
    DeleteUploadsRequest,
    DeleteUploadsResponse,
    GenerateRequest,
    GenerateResponse,
    GenerationRunLog,
    IngestResponse,
    KnowledgeBaseStatus,
    RebuildCorpusResponse,
    ResetCorpusRequest,
    ResetCorpusResponse,
    UploadedFileInfo,
    UploadListResponse,
)
from backend.tools.llm_client import LLMClient
from backend.tools.vector_store import FAISSVectorStore
from ingestion.pipeline import ingest_file

logger = get_logger(__name__)

SUPPORTED_UPLOAD_EXTS = {".pdf", ".md", ".markdown", ".txt", ".text"}
SUPPORTED_ARTIFACT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_GENERATION_LOGS = 100


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
    return safe or "document"


def _display_name_from_stored(stored_name: str) -> str:
    if "__" in stored_name:
        return stored_name.split("__", 1)[1]
    return stored_name


def _list_upload_files(cfg, vector_store: FAISSVectorStore) -> list[UploadedFileInfo]:
    chunk_count_by_file: dict[str, int] = {}
    for chunk in vector_store.all_chunks():
        chunk_count_by_file[chunk.filename] = (
            chunk_count_by_file.get(chunk.filename, 0) + 1
        )

    files: list[UploadedFileInfo] = []
    cfg.uploads_path.mkdir(parents=True, exist_ok=True)
    for fp in sorted(
        cfg.uploads_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if not fp.is_file() or fp.suffix.lower() not in SUPPORTED_UPLOAD_EXTS:
            continue
        stat = fp.stat()
        chunk_count = chunk_count_by_file.get(fp.name, 0)
        files.append(
            UploadedFileInfo(
                stored_name=fp.name,
                display_name=_display_name_from_stored(fp.name),
                size_bytes=stat.st_size,
                uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                chunk_count=chunk_count,
                indexed=chunk_count > 0,
            )
        )

    return files


def _rebuild_index_from_uploads(cfg, vector_store: FAISSVectorStore) -> tuple[int, int]:
    cfg.uploads_path.mkdir(parents=True, exist_ok=True)
    vector_store.clear()

    uploads = [
        fp
        for fp in sorted(cfg.uploads_path.iterdir())
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_UPLOAD_EXTS
    ]

    total_docs = 0
    total_chunks = 0
    for fp in uploads:
        try:
            chunks = ingest_file(fp)
            total_chunks += vector_store.add_chunks(chunks)
            total_docs += 1
        except Exception as e:
            logger.error("Failed to rebuild file", file=str(fp), error=str(e))

    if total_chunks > 0:
        vector_store.save()

    return total_docs, total_chunks


def _rebuild_if_uploads_exist(
    cfg, vector_store: FAISSVectorStore
) -> tuple[bool, int, int]:
    """
    Rebuild the FAISS index from uploaded files only when the index is empty.
    Returns: (rebuilt, documents_processed, chunks_indexed)
    """
    if vector_store.total_vectors > 0:
        return False, 0, 0

    cfg.uploads_path.mkdir(parents=True, exist_ok=True)
    has_uploads = any(
        fp.is_file() and fp.suffix.lower() in SUPPORTED_UPLOAD_EXTS
        for fp in cfg.uploads_path.iterdir()
    )
    if not has_uploads:
        return False, 0, 0

    docs, chunks = _rebuild_index_from_uploads(cfg, vector_store)
    return chunks > 0, docs, chunks


def _append_generation_log(
    logs: list[GenerationRunLog], entry: GenerationRunLog
) -> None:
    logs.insert(0, entry)
    if len(logs) > MAX_GENERATION_LOGS:
        del logs[MAX_GENERATION_LOGS:]


class InMemoryRateLimiter:
    """Simple per-key sliding-window limiter for abuse control."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, max_requests: int, window_seconds: int) -> int | None:
        now = time.monotonic()
        q = self._hits.setdefault(key, deque())
        cutoff = now - window_seconds

        while q and q[0] <= cutoff:
            q.popleft()

        if len(q) >= max_requests:
            retry_after = max(1, int(window_seconds - (now - q[0])))
            return retry_after

        q.append(now)
        return None


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",", 1)[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


# ── Application Factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    cfg = get_settings()
    setup_logging(cfg.log_level, cfg.environment)

    if cfg.environment == "production":
        if any(origin.strip() == "*" for origin in cfg.cors_origins):
            raise RuntimeError("CORS_ORIGINS cannot contain '*' in production")

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
    _generation_logs: list[GenerationRunLog] = []
    _rate_limiter = InMemoryRateLimiter()

    def enforce_rate_limit(request: Request, scope: str, limit: int) -> None:
        key = f"{scope}:{_client_ip(request)}"
        retry_after = _rate_limiter.check(
            key=key,
            max_requests=limit,
            window_seconds=cfg.rate_limit_window_seconds,
        )
        if retry_after is not None:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for '{scope}'. Retry later.",
                headers={"Retry-After": str(retry_after)},
            )

    # ── Security ──────────────────────────────────────────────────────────────
    security = HTTPBearer(auto_error=False)

    def verify_api_key(
        credentials: HTTPAuthorizationCredentials | None = Security(security),
    ) -> None:
        api_key = cfg.backend_api_key.strip()

        if cfg.environment == "production" and not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server misconfigured: BACKEND_API_KEY is required in production",
            )

        if not api_key:
            return

        provided = credentials.credentials if credentials else ""
        if not provided or not secrets.compare_digest(provided, api_key):
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

    @app.get("/api/uploads", response_model=UploadListResponse)
    async def list_uploads(_: None = Depends(verify_api_key)):
        files = _list_upload_files(cfg, _vector_store)
        return UploadListResponse(
            total_files=len(files),
            total_size_bytes=sum(f.size_bytes for f in files),
            files=files,
        )

    @app.post("/api/uploads/delete", response_model=DeleteUploadsResponse)
    async def delete_uploads(
        request: DeleteUploadsRequest,
        http_request: Request,
        _: None = Depends(verify_api_key),
    ):
        enforce_rate_limit(
            request=http_request,
            scope="management",
            limit=cfg.rate_limit_management_requests,
        )

        if not request.stored_names:
            return DeleteUploadsResponse()

        deleted: list[str] = []
        not_found: list[str] = []

        for raw_name in request.stored_names:
            name = Path(raw_name).name
            if name != raw_name:
                not_found.append(raw_name)
                continue

            fp = cfg.uploads_path / name
            if fp.exists() and fp.is_file():
                fp.unlink()
                deleted.append(name)
            else:
                not_found.append(name)

        rebuild_docs, rebuild_chunks = await run_in_threadpool(
            _rebuild_index_from_uploads, cfg, _vector_store
        )

        return DeleteUploadsResponse(
            deleted=deleted,
            not_found=not_found,
            rebuild_documents_processed=rebuild_docs,
            rebuild_chunks_indexed=rebuild_chunks,
        )

    @app.post("/api/kb/reset", response_model=ResetCorpusResponse)
    async def reset_corpus(
        request: ResetCorpusRequest,
        http_request: Request,
        _: None = Depends(verify_api_key),
    ):
        enforce_rate_limit(
            request=http_request,
            scope="management",
            limit=cfg.rate_limit_management_requests,
        )

        _vector_store.clear()
        _articles.clear()
        _generation_logs.clear()

        uploads_removed = 0
        if request.delete_uploads:
            cfg.uploads_path.mkdir(parents=True, exist_ok=True)
            for fp in cfg.uploads_path.iterdir():
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_UPLOAD_EXTS:
                    fp.unlink()
                    uploads_removed += 1

        artifacts_removed = 0
        if request.delete_artifacts:
            cfg.artifacts_path.mkdir(parents=True, exist_ok=True)
            for fp in cfg.artifacts_path.iterdir():
                if fp.is_file():
                    fp.unlink()
                    artifacts_removed += 1

        return ResetCorpusResponse(
            status="reset",
            uploads_removed=uploads_removed,
            artifacts_removed=artifacts_removed,
        )

    @app.post("/api/kb/rebuild", response_model=RebuildCorpusResponse)
    async def rebuild_corpus(
        http_request: Request,
        _: None = Depends(verify_api_key),
    ):
        enforce_rate_limit(
            request=http_request,
            scope="management",
            limit=cfg.rate_limit_management_requests,
        )

        docs, chunks = await run_in_threadpool(
            _rebuild_index_from_uploads, cfg, _vector_store
        )
        return RebuildCorpusResponse(
            status="rebuilt" if chunks > 0 else "empty",
            documents_processed=docs,
            chunks_indexed=chunks,
        )

    @app.post("/api/ingest", response_model=IngestResponse)
    async def ingest_documents(
        http_request: Request,
        files: list[UploadFile] = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        _: None = Depends(verify_api_key),
    ):
        enforce_rate_limit(
            request=http_request,
            scope="ingest",
            limit=cfg.rate_limit_ingest_requests,
        )

        if len(files) > cfg.max_upload_files_per_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files in one request. Max allowed: {cfg.max_upload_files_per_request}",
            )

        job_id = str(uuid.uuid4())
        logger.info("Ingest endpoint hit", job_id=job_id, files_count=len(files))
        t0 = time.perf_counter()

        saved_paths: list[Path] = []

        try:
            for upload in files:
                original_name = Path(upload.filename or "file.txt").name
                suffix = Path(original_name).suffix.lower()
                if suffix not in SUPPORTED_UPLOAD_EXTS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type '{suffix}'. Allowed: {SUPPORTED_UPLOAD_EXTS}",
                    )

                safe_name = _sanitize_filename(original_name)
                if Path(safe_name).suffix.lower() != suffix and suffix:
                    safe_name = f"{Path(safe_name).stem}{suffix}"

                dest = cfg.uploads_path / f"{uuid.uuid4().hex}__{safe_name}"
                content = await upload.read()
                max_size_bytes = cfg.max_upload_file_size_mb * 1024 * 1024
                if len(content) > max_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File '{original_name}' exceeds max upload size "
                            f"of {cfg.max_upload_file_size_mb} MB"
                        ),
                    )

                dest.write_bytes(content)
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
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate_article(
        request: GenerateRequest,
        http_request: Request,
        _: None = Depends(verify_api_key),
    ):
        enforce_rate_limit(
            request=http_request,
            scope="generate",
            limit=cfg.rate_limit_generate_requests,
        )

        if _vector_store.total_vectors == 0:
            rebuilt, docs, chunks = await run_in_threadpool(
                _rebuild_if_uploads_exist, cfg, _vector_store
            )
            if rebuilt:
                logger.info(
                    "Rebuilt index from existing uploads before generation",
                    documents_processed=docs,
                    chunks_indexed=chunks,
                )

        if _vector_store.total_vectors == 0:
            raise HTTPException(
                status_code=422,
                detail="Knowledge base is empty. Please ingest documents first via POST /api/ingest",
            )

        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()

        try:
            # Add timeout to prevent indefinite hangs (LLM API timeouts, etc)
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
                    timeout=300.0,  # 5 minute timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Generation pipeline timed out",
                    timeout_seconds=300,
                )
                elapsed = time.perf_counter() - t0
                _append_generation_log(
                    _generation_logs,
                    GenerationRunLog(
                        run_id=run_id,
                        topic=request.topic,
                        status="failed",
                        started_at=started_at,
                        duration_seconds=round(elapsed, 3),
                        generate_images=request.generate_images,
                        run_qa=request.run_qa,
                        error="Generation request timed out after 5 minutes",
                    ),
                )
                raise HTTPException(
                    status_code=504,
                    detail="Generation request timed out after 5 minutes",
                )
        except HTTPException:
            raise
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("Generation pipeline failed", error=str(e))
            _append_generation_log(
                _generation_logs,
                GenerationRunLog(
                    run_id=run_id,
                    topic=request.topic,
                    status="failed",
                    started_at=started_at,
                    duration_seconds=round(elapsed, 3),
                    generate_images=request.generate_images,
                    run_qa=request.run_qa,
                    error=str(e),
                ),
            )
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

        if ctx.article:
            _articles[ctx.article.article_id] = ctx.article.model_dump()

        elapsed = time.perf_counter() - t0
        image_attempted = 0
        if request.generate_images and ctx.outline:
            image_attempted = 1 + min(3, len(ctx.outline.sections))
        image_generated = sum(1 for url in ctx.image_urls.values() if url)
        image_failed = max(0, image_attempted - image_generated)

        _append_generation_log(
            _generation_logs,
            GenerationRunLog(
                run_id=run_id,
                topic=request.topic,
                status="completed" if ctx.article else "failed",
                started_at=started_at,
                duration_seconds=round(elapsed, 3),
                generate_images=request.generate_images,
                run_qa=request.run_qa,
                stage_timings=ctx.stage_timings,
                token_usage=ctx.article.token_usage if ctx.article else {},
                image_attempted=image_attempted,
                image_generated=image_generated,
                image_failed=image_failed,
                qa_passed=ctx.qa_report.passed if ctx.qa_report else None,
                qa_overall_confidence=(
                    ctx.qa_report.overall_confidence if ctx.qa_report else None
                ),
                qa_grounding_score=(
                    ctx.qa_report.grounding_score if ctx.qa_report else None
                ),
                qa_readability_score=(
                    ctx.qa_report.readability_score if ctx.qa_report else None
                ),
                qa_warning_count=len(ctx.qa_report.warnings) if ctx.qa_report else 0,
                error=None if ctx.article else "Pipeline returned no article",
            ),
        )

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

    @app.get("/api/dashboard/logs", response_model=list[GenerationRunLog])
    async def dashboard_logs(_: None = Depends(verify_api_key)):
        return _generation_logs[:50]

    @app.get("/api/dashboard/metrics", response_model=DashboardMetrics)
    async def dashboard_metrics(_: None = Depends(verify_api_key)):
        total_runs = len(_generation_logs)
        successful_runs = sum(1 for r in _generation_logs if r.status == "completed")
        failed_runs = total_runs - successful_runs
        qa_enabled_runs = sum(1 for r in _generation_logs if r.run_qa)
        qa_passed_runs = sum(
            1 for r in _generation_logs if r.run_qa and r.qa_passed is True
        )
        qa_failed_runs = sum(
            1 for r in _generation_logs if r.run_qa and r.qa_passed is False
        )
        avg_duration = (
            sum(r.duration_seconds for r in _generation_logs) / total_runs
            if total_runs
            else 0.0
        )

        total_input_tokens = sum(
            int(r.token_usage.get("total_input_tokens", 0) or 0)
            for r in _generation_logs
        )
        total_output_tokens = sum(
            int(r.token_usage.get("total_output_tokens", 0) or 0)
            for r in _generation_logs
        )

        return DashboardMetrics(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            qa_enabled_runs=qa_enabled_runs,
            qa_passed_runs=qa_passed_runs,
            qa_failed_runs=qa_failed_runs,
            avg_duration_seconds=round(avg_duration, 3),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            recent_runs=_generation_logs[:20],
        )

    @app.get("/artifacts/{filename}")
    async def serve_artifact(filename: str):
        """Serve generated images."""
        safe_name = Path(filename).name
        if safe_name != filename:
            raise HTTPException(status_code=400, detail="Invalid artifact name")

        if Path(safe_name).suffix.lower() not in SUPPORTED_ARTIFACT_EXTS:
            raise HTTPException(status_code=404, detail="Artifact not found")

        fp = cfg.artifacts_path / safe_name
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(fp, headers={"Cache-Control": "public, max-age=86400"})

    @app.delete("/api/kb/clear")
    async def clear_knowledge_base(_: None = Depends(verify_api_key)):
        _vector_store.clear()
        _articles.clear()
        _generation_logs.clear()
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

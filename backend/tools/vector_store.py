"""
KG-MAG — FAISS Vector Store
============================
Persistent, production-ready FAISS index for semantic search.

Architecture
------------
- Index type: IndexFlatIP (inner product, works with L2-normalized vectors = cosine sim)
- We store chunks separately in a JSON sidecar file; FAISS only stores vectors.
- IndexIDMap2 lets us map FAISS integer IDs ↔ our string chunk_ids.
- Thread-safe reads via a read lock; writes acquire an exclusive lock.

Why FAISS over Chroma for default?
------------------------------------
- No external service to run (Chroma needs a separate container)
- Instant startup — great for local dev and CI
- Handles millions of vectors on CPU with excellent recall
"""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog

from backend.core.config import get_settings
from backend.core.models import DocumentChunk, RetrievedChunk
from backend.tools.embedding import EmbeddingEngine

logger = structlog.get_logger(__name__)


class FAISSVectorStore:
    """
    Persistent FAISS-based vector store.

    Persists to disk as:
        <index_path>/vectors.index   — FAISS binary index
        <index_path>/chunks.pkl      — mapping of int_id → DocumentChunk
        <index_path>/meta.json       — index statistics
    """

    def __init__(self, index_path: Optional[Path] = None) -> None:
        import faiss  # type: ignore  # lazy import; fails clearly if not installed

        self._faiss = faiss
        cfg = get_settings()
        self._index_path = index_path or cfg.faiss_index_path
        self._index_path.mkdir(parents=True, exist_ok=True)

        self._engine = EmbeddingEngine()
        self._dim = self._engine.dimension

        self._lock = threading.RLock()
        self._chunks: dict[int, DocumentChunk] = {}   # int_id → chunk
        self._str_to_int: dict[str, int] = {}         # chunk_id → int_id
        self._next_id: int = 0
        self._index: "faiss.IndexIDMap2" = self._empty_index()

        # Attempt to load persisted index
        if self._vectors_file.exists() and self._chunks_file.exists():
            self._load()

    # ── Paths ─────────────────────────────────────────────────────────────────

    @property
    def _vectors_file(self) -> Path:
        return self._index_path / "vectors.index"

    @property
    def _chunks_file(self) -> Path:
        return self._index_path / "chunks.pkl"

    @property
    def _meta_file(self) -> Path:
        return self._index_path / "meta.json"

    # ── Construction helpers ──────────────────────────────────────────────────

    def _empty_index(self) -> Any:
        flat = self._faiss.IndexFlatIP(self._dim)
        return self._faiss.IndexIDMap2(flat)

    # ── Core operations ───────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Embed and add chunks to the index.
        Returns the number of new vectors added.
        """
        if not chunks:
            return 0

        # Skip already-indexed chunks (idempotent)
        new_chunks = [c for c in chunks if c.chunk_id not in self._str_to_int]
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        vectors = self._engine.encode(texts, normalize=True, show_progress=len(texts) > 100)

        with self._lock:
            int_ids = np.arange(
                self._next_id, self._next_id + len(new_chunks), dtype=np.int64
            )
            self._index.add_with_ids(vectors, int_ids)

            for chunk, iid in zip(new_chunks, int_ids):
                self._chunks[int(iid)] = chunk
                self._str_to_int[chunk.chunk_id] = int(iid)

            self._next_id += len(new_chunks)

        logger.info("Added chunks to index", added=len(new_chunks), total=self._next_id)
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most similar chunks for a query string.
        Returns RetrievedChunk objects sorted by descending similarity.
        """
        cfg = get_settings()
        k = top_k or cfg.top_k_retrieval

        if self._index.ntotal == 0:
            logger.warning("Empty index — no results")
            return []

        query_vec = self._engine.encode([query], normalize=True)
        k = min(k, self._index.ntotal)

        with self._lock:
            scores, ids = self._index.search(query_vec, k)

        results: list[RetrievedChunk] = []
        for rank, (score, iid) in enumerate(zip(scores[0], ids[0]), start=1):
            if iid == -1:    # FAISS padding
                continue
            chunk = self._chunks.get(int(iid))
            if chunk is None:
                continue
            # Clip to [0, 1] — inner product on normalized vectors can slightly exceed 1
            results.append(
                RetrievedChunk(chunk=chunk, score=float(np.clip(score, 0.0, 1.0)), rank=rank)
            )

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index and chunk metadata to disk."""
        with self._lock:
            self._faiss.write_index(self._index, str(self._vectors_file))
            with open(self._chunks_file, "wb") as f:
                pickle.dump(
                    {"chunks": self._chunks, "str_to_int": self._str_to_int, "next_id": self._next_id},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            self._meta_file.write_text(
                json.dumps({"total_vectors": self._next_id, "dim": self._dim})
            )
        logger.info("Index saved", path=str(self._index_path), vectors=self._next_id)

    def _load(self) -> None:
        """Load a previously saved index from disk."""
        try:
            self._index = self._faiss.read_index(str(self._vectors_file))
            with open(self._chunks_file, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._str_to_int = data["str_to_int"]
            self._next_id = data["next_id"]
            logger.info(
                "Loaded existing FAISS index",
                vectors=self._next_id,
                path=str(self._index_path),
            )
        except Exception as e:
            logger.error("Failed to load index — starting fresh", error=str(e))
            self._reset()

    def _reset(self) -> None:
        self._chunks = {}
        self._str_to_int = {}
        self._next_id = 0
        self._index = self._empty_index()

    def clear(self) -> None:
        """Delete all indexed data."""
        with self._lock:
            self._reset()
            for f in [self._vectors_file, self._chunks_file, self._meta_file]:
                if f.exists():
                    f.unlink()
        logger.info("Index cleared")

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        iid = self._str_to_int.get(chunk_id)
        if iid is None:
            return None
        return self._chunks.get(iid)

    def all_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks.values())

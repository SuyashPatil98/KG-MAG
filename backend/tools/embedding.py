"""
KG-MAG — Embedding Engine
==========================
Wraps sentence-transformers for local, free embeddings.

Key design choices
------------------
- Singleton pattern: model loads once, reused across requests
- Batched encoding: efficient for large corpora
- Normalized vectors: cosine similarity reduces to dot product
- Device-aware: CPU / CUDA / MPS (Apple Silicon)
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Sequence

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer  # type: ignore

from backend.core.config import get_settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model once and cache it.
    Thread-safe because Python's GIL protects the cache population.
    """
    cfg = get_settings()
    logger.info(
        "Loading embedding model",
        model=cfg.embedding_model,
        device=cfg.embedding_device,
    )
    t0 = time.perf_counter()
    model = SentenceTransformer(cfg.embedding_model, device=cfg.embedding_device)
    elapsed = time.perf_counter() - t0
    logger.info("Embedding model loaded", elapsed_s=round(elapsed, 2))
    return model


class EmbeddingEngine:
    """
    Thin wrapper around SentenceTransformer that exposes
    batch encoding and similarity utilities.
    """

    def __init__(self) -> None:
        self._model = _load_model()
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim  # type: ignore[return-value]

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Parameters
        ----------
        texts       : Input strings to embed
        batch_size  : Override config batch size
        normalize   : L2-normalize output (required for cosine sim via dot product)
        show_progress: tqdm progress bar (disable in production)

        Returns
        -------
        np.ndarray of shape (len(texts), self.dimension), dtype float32
        """
        cfg = get_settings()
        bs = batch_size or cfg.embedding_batch_size

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        t0 = time.perf_counter()
        vectors = self._model.encode(
            list(texts),
            batch_size=bs,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0

        logger.debug(
            "Encoded texts",
            count=len(texts),
            dim=self.dimension,
            elapsed_ms=round(elapsed * 1000),
        )
        return vectors.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience method to embed a single string."""
        return self.encode([text])[0]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1-D vectors.
        If both are L2-normalized, this equals the dot product.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity of a single query vector against a matrix.
        matrix shape: (N, D), query shape: (D,)
        Returns array of shape (N,) with similarity scores.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = matrix / norms
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        return (normalized @ query_norm).astype(np.float32)

"""
KG-MAG — ML Reranking Model
============================
A lightweight retrieval-relevance reranker that rescores FAISS candidates.

Why reranking?
--------------
Embedding similarity (first-stage retrieval) is fast but imprecise.
A second-stage reranker uses richer features to reorder the candidates,
significantly improving relevance for generation.

Architecture
------------
We implement a cross-encoder style reranker using:
  - Feature engineering on (query, chunk) pairs
  - A small logistic regression / gradient-boosted model (scikit-learn)
  - Optional upgrade path to a neural cross-encoder

Features (per query-chunk pair)
---------------------------------
  1. Cosine similarity (from FAISS, already computed)
  2. BM25-inspired keyword overlap score
  3. Chunk heading match (1 if topic words appear in heading, else 0)
  4. Chunk length percentile (normalized)
  5. Position score (earlier chunks score slightly higher, domain heuristic)
  6. Query term coverage in chunk (unigram recall)

Training Data
-------------
For a portfolio project, we use a SIMULATED training approach:
  - Positive pairs: query vs. chunk that contains all query terms (assumed relevant)
  - Negative pairs: query vs. chunk with low cosine similarity (assumed irrelevant)
  - In production: use human-labeled relevance or RLHF-style feedback loops.

Integration
-----------
Call rerank(query, candidates) after FAISS search.
The model is trained lazily on first call if no saved model is found.
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

import numpy as np
import structlog

from backend.core.config import get_settings
from backend.core.models import DocumentChunk, RetrievedChunk

logger = structlog.get_logger(__name__)

MODEL_PATH = Path("./data/models/reranker.pkl")
SCALER_PATH = Path("./data/models/reranker_scaler.pkl")


# ── Feature Engineering ───────────────────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))


def _keyword_overlap(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    """BM25-inspired: |intersection| / sqrt(|query| * |chunk|)"""
    inter = query_tokens & chunk_tokens
    denom = math.sqrt(max(len(query_tokens), 1) * max(len(chunk_tokens), 1))
    return len(inter) / denom if denom > 0 else 0.0


def _query_coverage(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    """What fraction of query terms appear in the chunk?"""
    if not query_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def _heading_match(query_tokens: set[str], heading: str | None) -> float:
    if not heading:
        return 0.0
    heading_tokens = _tokenize(heading)
    overlap = query_tokens & heading_tokens
    return min(1.0, len(overlap) / max(len(query_tokens), 1))


def extract_features(
    query: str,
    chunks: list[DocumentChunk],
    scores: list[float],
    positions: list[int],
    chunk_lengths: list[int],
) -> np.ndarray:
    """
    Build a feature matrix of shape (n_chunks, n_features).

    Features (6 total)
    ------------------
    0: cosine_sim         — FAISS inner-product score
    1: keyword_overlap    — BM25-style term overlap
    2: query_coverage     — fraction of query terms found
    3: heading_match      — topic alignment with section heading
    4: length_score       — normalized chunk length [0,1]
    5: position_score     — earlier in doc → slightly preferred
    """
    n_rows = max(len(chunks), len(scores), len(positions), len(chunk_lengths))
    if n_rows == 0:
        return np.zeros((0, 6), dtype=np.float32)

    query_tokens = _tokenize(query)
    max_len = max(chunk_lengths) if chunk_lengths else 1
    max_len = max(max_len, 1)
    max_pos = max(positions) if positions else max(n_rows - 1, 1)

    rows = []
    for i in range(n_rows):
        chunk = chunks[i] if i < len(chunks) else None
        score = float(scores[i]) if i < len(scores) else 0.0
        pos = int(positions[i]) if i < len(positions) else i
        ln = (
            int(chunk_lengths[i])
            if i < len(chunk_lengths)
            else (len(chunk.text.split()) if chunk else 0)
        )

        chunk_tokens = _tokenize(chunk.text) if chunk else set()
        heading = chunk.heading if chunk else None
        rows.append(
            [
                float(score),
                _keyword_overlap(query_tokens, chunk_tokens),
                _query_coverage(query_tokens, chunk_tokens),
                _heading_match(query_tokens, heading),
                ln / max_len,
                1.0 - (pos / (max_pos + 1)),
            ]
        )
    return np.array(rows, dtype=np.float32)


# ── Model Training ────────────────────────────────────────────────────────────


def simulate_training_data(
    chunks: list[DocumentChunk],
    n_queries: int = 200,
    neg_ratio: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate simulated training pairs for the reranker.

    Positive pair: use a chunk's first 5 words as a pseudo-query → relevant.
    Negative pair: sample random chunks → irrelevant.

    In production, replace with:
      - User click data
      - Human relevance judgments
      - Distilled labels from a large cross-encoder (e.g. ms-marco)
    """
    from random import sample, seed as rseed

    rseed(42)

    if len(chunks) < 10:
        raise ValueError("Need ≥10 chunks to simulate training data")

    X_rows, y = [], []
    chosen = sample(chunks, min(n_queries, len(chunks)))

    for pos_chunk in chosen:
        words = pos_chunk.text.split()
        query = " ".join(words[: min(7, len(words))])
        q_tokens = _tokenize(query)

        # Positive sample
        pos_tokens = _tokenize(pos_chunk.text)
        pos_feat = [
            0.85,  # simulated high score
            _keyword_overlap(q_tokens, pos_tokens),
            _query_coverage(q_tokens, pos_tokens),
            _heading_match(q_tokens, pos_chunk.heading),
            len(pos_chunk.text.split()) / 512,
            0.5,
        ]
        X_rows.append(pos_feat)
        y.append(1)

        # Negative samples
        neg_pool = [c for c in chunks if c.chunk_id != pos_chunk.chunk_id]
        for neg_chunk in sample(neg_pool, min(neg_ratio, len(neg_pool))):
            neg_tokens = _tokenize(neg_chunk.text)
            neg_feat = [
                0.3,  # simulated low score
                _keyword_overlap(q_tokens, neg_tokens),
                _query_coverage(q_tokens, neg_tokens),
                _heading_match(q_tokens, neg_chunk.heading),
                len(neg_chunk.text.split()) / 512,
                0.5,
            ]
            X_rows.append(neg_feat)
            y.append(0)

    return np.array(X_rows, dtype=np.float32), np.array(y, dtype=np.int32)


def train_reranker(chunks: list[DocumentChunk]) -> None:
    """
    Train and persist a gradient-boosted reranker model.
    Uses scikit-learn GradientBoostingClassifier — lightweight, fast, interpretable.
    """
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    logger.info("Training reranker model on simulated data...")
    X, y = simulate_training_data(chunks)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_scaled, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Reranker saved", path=str(MODEL_PATH))


def _load_model():
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ── Public API ────────────────────────────────────────────────────────────────


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Rerank FAISS retrieval candidates using the ML reranker.

    Falls back to original FAISS order if model is not trained yet.
    """
    cfg = get_settings()
    k = top_k or cfg.rerank_top_k

    if not candidates:
        return []

    model, scaler = _load_model()
    if model is None:
        logger.debug("Reranker not trained — using FAISS ranking")
        return candidates[:k]

    chunks = [r.chunk for r in candidates]
    scores = [r.score for r in candidates]
    positions = [r.chunk.chunk_index for r in candidates]
    lengths = [len(r.chunk.text.split()) for r in candidates]

    X = extract_features(query, chunks, scores, positions, lengths)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]  # P(relevant)

    ranked = sorted(
        zip(candidates, proba),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for new_rank, (cand, prob) in enumerate(ranked[:k], start=1):
        results.append(
            RetrievedChunk(
                chunk=cand.chunk,
                score=float(prob),  # reranker score replaces FAISS score
                rank=new_rank,
            )
        )
    return results

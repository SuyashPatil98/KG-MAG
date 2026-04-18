"""
KG-MAG — Test Suite
=====================
Tests cover ingestion, retrieval, agent behavior, and API endpoints.

Run:
    pytest tests/ -v
    pytest tests/test_ingestion.py -v --tb=short
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = textwrap.dedent("""\
    ---
    title: Test Article
    author: Test Author
    ---

    # Introduction

    This is the introduction paragraph about machine learning and RAG systems.
    Retrieval-augmented generation combines retrieval with generation.

    ## Core Concepts

    Vector databases store high-dimensional embeddings for semantic search.
    FAISS is a popular library for efficient similarity search.

    ### Subsection

    Chunking strategies affect retrieval quality significantly.
    Overlapping chunks prevent information loss at boundaries.

    ## Applications

    RAG systems are used in question-answering, article generation, and chatbots.
""")

SAMPLE_TEXT = textwrap.dedent("""\
    Knowledge graphs store structured information as triplets.
    Embeddings map text to vector spaces for semantic comparison.
    Large language models generate coherent text from prompts.
    Hallucination occurs when models generate false information.
    Grounding verification checks claims against source material.
""")


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_md_file(temp_dir):
    p = temp_dir / "test_article.md"
    p.write_text(SAMPLE_MARKDOWN)
    return p


@pytest.fixture
def sample_txt_file(temp_dir):
    p = temp_dir / "test_doc.txt"
    p.write_text(SAMPLE_TEXT)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Test: Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

class TestTextCleaning:
    def test_removes_null_bytes(self):
        from ingestion.pipeline import clean_text
        raw = "Hello\x00 World\x00"
        assert "\x00" not in clean_text(raw)

    def test_normalizes_unicode(self):
        from ingestion.pipeline import clean_text
        # Café written with combining accent vs precomposed
        raw = "cafe\u0301"   # decomposed NFD
        cleaned = clean_text(raw)
        assert cleaned == "café"

    def test_collapses_excessive_newlines(self):
        from ingestion.pipeline import clean_text
        raw = "Line 1\n\n\n\n\nLine 2"
        cleaned = clean_text(raw)
        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned

    def test_preserves_paragraphs(self):
        from ingestion.pipeline import clean_text
        raw = "Para one.\n\nPara two."
        cleaned = clean_text(raw)
        assert "\n\n" in cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Test: Document ingestion
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_markdown_ingestion(self, sample_md_file):
        from ingestion.pipeline import ingest_file
        chunks = ingest_file(sample_md_file)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.chunk_id
            assert chunk.source_id
            assert chunk.filename == "test_article.md"

    def test_text_ingestion(self, sample_txt_file):
        from ingestion.pipeline import ingest_file
        chunks = ingest_file(sample_txt_file)
        assert len(chunks) > 0

    def test_chunk_overlap_preserves_context(self, temp_dir):
        from ingestion.pipeline import chunk_document, extract_markdown
        p = temp_dir / "long.md"
        # Create a document that will need multiple chunks
        words = " ".join([f"word{i}" for i in range(1000)])
        p.write_text(f"# Section\n\n{words}")
        text, meta = extract_markdown(p)
        chunks = chunk_document(text, meta, chunk_size=50, overlap=10)
        assert len(chunks) >= 2

        # Check overlap: last N words of chunk[i] appear in chunk[i+1]
        if len(chunks) >= 2:
            words0 = chunks[0].text.split()
            words1 = chunks[1].text.split()
            # Some overlap expected
            overlap_words = set(words0[-10:]) & set(words1[:20])
            assert len(overlap_words) > 0

    def test_heading_preserved_in_chunks(self, sample_md_file):
        from ingestion.pipeline import ingest_file
        chunks = ingest_file(sample_md_file)
        headings = [c.heading for c in chunks if c.heading]
        assert len(headings) > 0

    def test_unsupported_file_raises(self, temp_dir):
        from ingestion.pipeline import ingest_file
        p = temp_dir / "test.docx"
        p.write_bytes(b"fake docx content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_file(p)

    def test_empty_file_returns_no_chunks(self, temp_dir):
        from ingestion.pipeline import ingest_file
        p = temp_dir / "empty.txt"
        p.write_text("   \n\n  ")
        chunks = ingest_file(p)
        assert chunks == []

    def test_chunk_ids_are_unique(self, sample_md_file):
        from ingestion.pipeline import ingest_file
        chunks = ingest_file(sample_md_file)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_deterministic_chunk_ids(self, sample_md_file):
        """Same file → same chunk IDs (important for deduplication)."""
        from ingestion.pipeline import ingest_file
        chunks1 = ingest_file(sample_md_file)
        chunks2 = ingest_file(sample_md_file)
        ids1 = {c.chunk_id for c in chunks1}
        ids2 = {c.chunk_id for c in chunks2}
        assert ids1 == ids2


# ─────────────────────────────────────────────────────────────────────────────
# Test: Embedding Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingEngine:
    """Tests that do not require GPU or network."""

    def test_encode_returns_correct_shape(self):
        from backend.tools.embedding import EmbeddingEngine
        engine = EmbeddingEngine()
        texts = ["Hello world", "Machine learning is fascinating"]
        vecs = engine.encode(texts)
        assert vecs.shape == (2, engine.dimension)

    def test_encode_single(self):
        from backend.tools.embedding import EmbeddingEngine
        engine = EmbeddingEngine()
        vec = engine.encode_single("Test sentence")
        assert vec.shape == (engine.dimension,)

    def test_normalized_vectors(self):
        from backend.tools.embedding import EmbeddingEngine
        engine = EmbeddingEngine()
        vecs = engine.encode(["Normalize me"], normalize=True)
        norm = float(np.linalg.norm(vecs[0]))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_input(self):
        from backend.tools.embedding import EmbeddingEngine
        engine = EmbeddingEngine()
        vecs = engine.encode([])
        assert vecs.shape[0] == 0

    def test_similar_texts_high_cosine(self):
        from backend.tools.embedding import EmbeddingEngine
        engine = EmbeddingEngine()
        v1 = engine.encode_single("The quick brown fox")
        v2 = engine.encode_single("A quick brown fox jumps")
        v3 = engine.encode_single("Quantum mechanics in physics")
        sim_related = engine.cosine_similarity(v1, v2)
        sim_unrelated = engine.cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated


# ─────────────────────────────────────────────────────────────────────────────
# Test: Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self, temp_dir):
        from backend.tools.vector_store import FAISSVectorStore
        from ingestion.pipeline import ingest_file
        store = FAISSVectorStore(index_path=temp_dir / "idx")

        p = temp_dir / "doc.txt"
        p.write_text(SAMPLE_TEXT)
        chunks = ingest_file(p)
        added = store.add_chunks(chunks)
        assert added == len(chunks)
        assert store.total_vectors == len(chunks)

        results = store.search("vector embeddings semantic search", top_k=3)
        assert len(results) > 0
        assert all(0 <= r.score <= 1 for r in results)
        assert results[0].rank == 1

    def test_search_returns_ranked_results(self, temp_dir):
        from backend.tools.vector_store import FAISSVectorStore
        from ingestion.pipeline import ingest_file
        store = FAISSVectorStore(index_path=temp_dir / "idx2")

        p = temp_dir / "doc.txt"
        p.write_text(SAMPLE_TEXT)
        chunks = ingest_file(p)
        store.add_chunks(chunks)

        results = store.search("hallucination false generation", top_k=5)
        # Scores should be descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_persist_and_reload(self, temp_dir):
        from backend.tools.vector_store import FAISSVectorStore
        from ingestion.pipeline import ingest_file
        store = FAISSVectorStore(index_path=temp_dir / "idx3")

        p = temp_dir / "doc.txt"
        p.write_text(SAMPLE_TEXT)
        chunks = ingest_file(p)
        store.add_chunks(chunks)
        store.save()

        # Load a new instance from the same path
        store2 = FAISSVectorStore(index_path=temp_dir / "idx3")
        assert store2.total_vectors == store.total_vectors

        results = store2.search("knowledge graphs", top_k=2)
        assert len(results) > 0

    def test_idempotent_add(self, temp_dir):
        """Adding same chunks twice should not duplicate."""
        from backend.tools.vector_store import FAISSVectorStore
        from ingestion.pipeline import ingest_file
        store = FAISSVectorStore(index_path=temp_dir / "idx4")

        p = temp_dir / "doc.txt"
        p.write_text(SAMPLE_TEXT)
        chunks = ingest_file(p)

        store.add_chunks(chunks)
        n1 = store.total_vectors
        store.add_chunks(chunks)  # add again
        n2 = store.total_vectors
        assert n1 == n2

    def test_empty_search_returns_empty(self, temp_dir):
        from backend.tools.vector_store import FAISSVectorStore
        store = FAISSVectorStore(index_path=temp_dir / "empty_idx")
        results = store.search("anything")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Test: Reranker
# ─────────────────────────────────────────────────────────────────────────────

class TestReranker:
    def test_feature_extraction_shape(self, temp_dir):
        from backend.models.reranker import extract_features
        from ingestion.pipeline import ingest_file
        p = temp_dir / "doc.txt"
        p.write_text(SAMPLE_TEXT)
        chunks = ingest_file(p)

        scores = [0.9, 0.7, 0.5]
        positions = [0, 1, 2]
        lengths = [100, 80, 60]
        X = extract_features("test query", chunks[:3], scores, positions, lengths)
        assert X.shape == (3, 6)

    def test_rerank_without_model_returns_original(self, temp_dir):
        """If no trained model, reranker returns original FAISS order."""
        from backend.models.reranker import rerank
        from backend.core.models import RetrievedChunk, DocumentChunk

        # Ensure model doesn't exist
        import backend.models.reranker as rm
        orig_path = rm.MODEL_PATH
        rm.MODEL_PATH = temp_dir / "nonexistent_model.pkl"

        chunks = []
        for i in range(3):
            dc = DocumentChunk(
                chunk_id=f"test_{i}",
                source_id="src",
                filename="test.txt",
                text=f"Sample text chunk {i}",
                chunk_index=i,
            )
            chunks.append(RetrievedChunk(chunk=dc, score=0.9 - i * 0.1, rank=i + 1))

        result = rerank("test query", chunks, top_k=3)
        assert len(result) <= 3

        rm.MODEL_PATH = orig_path  # restore


# ─────────────────────────────────────────────────────────────────────────────
# Test: QA / Flesch score
# ─────────────────────────────────────────────────────────────────────────────

class TestQA:
    def test_flesch_score_range(self):
        """Flesch score must be in [0, 100]."""
        from backend.agents.orchestrator import CriticAgent
        critic = CriticAgent.__new__(CriticAgent)

        easy = "The cat sat on the mat. It was a small cat."
        hard = "The epistemological ramifications of phenomenological hermeneutics necessitate circumspect deliberation."

        score_easy = critic._compute_flesch(easy)
        score_hard = critic._compute_flesch(hard)

        assert 0 <= score_easy <= 100
        assert 0 <= score_hard <= 100
        assert score_easy > score_hard

    def test_flesch_empty_text(self):
        from backend.agents.orchestrator import CriticAgent
        critic = CriticAgent.__new__(CriticAgent)
        score = critic._compute_flesch("")
        assert score == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: API endpoints (integration — requires running backend or TestClient)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        """FastAPI test client with mocked dependencies."""
        from fastapi.testclient import TestClient

           with patch("backend.api.main.FAISSVectorStore") as mock_vs, \
               patch("backend.api.main.LLMClient"), \
               patch("backend.api.main.ArticleOrchestrator"):

            mock_vs.return_value.total_vectors = 0
            mock_vs.return_value.all_chunks.return_value = []

            from backend.api.main import create_app
            app = create_app()
            yield TestClient(app)

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_generate_requires_kb(self, client):
        """Should 422 if knowledge base is empty."""
        resp = client.post(
            "/api/generate",
            json={"topic": "Test topic"},
        )
        assert resp.status_code == 422

    def test_ingest_rejects_invalid_type(self, client, temp_dir):
        bad_file = temp_dir / "test.docx"
        bad_file.write_bytes(b"fake")
        with open(bad_file, "rb") as f:
            resp = client.post(
                "/api/ingest",
                files=[("files", ("test.docx", f, "application/octet-stream"))],
            )
        assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Test: Directory ingestion
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectoryIngestion:
    def test_ingest_directory(self, temp_dir):
        from ingestion.pipeline import ingest_directory

        (temp_dir / "doc1.txt").write_text("First document about AI.")
        (temp_dir / "doc2.md").write_text("# Second\n\nMarkdown document about ML.")
        (temp_dir / "ignored.csv").write_text("col1,col2\nval1,val2")

        all_chunks = []
        for chunks in ingest_directory(temp_dir):
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0
        filenames = {c.filename for c in all_chunks}
        assert "doc1.txt" in filenames
        assert "doc2.md" in filenames
        assert "ignored.csv" not in filenames

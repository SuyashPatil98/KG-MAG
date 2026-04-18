# KG-MAG — Knowledge-Grounded Medium Article Generator

> **Build a production-grade RAG system that generates cited, hallucination-checked articles from your own documents.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Demo Video

[![Watch the demo](https://img.youtube.com/vi/CjslMkEpErk/hqdefault.jpg)](https://youtu.be/CjslMkEpErk)

End-to-end demo of KG-MAG ingestion, indexing, retrieval, and article generation.

---

## Table of Contents

1. [What Is KG-MAG?](#1-what-is-kg-mag)
2. [How the System Works](#2-how-the-system-works)
3. [Quick Start (5 minutes)](#3-quick-start)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Data Ingestion Pipeline](#5-data-ingestion-pipeline)
6. [Embedding & Retrieval](#6-embedding--retrieval)
7. [The Multi-Agent System](#7-the-multi-agent-system)
8. [ML Reranking Model](#8-ml-reranking-model)
9. [QA & Hallucination Detection](#9-qa--hallucination-detection)
10. [Image Generation](#10-image-generation)
11. [Docker Deployment](#11-docker-deployment)
12. [Vercel Deployment](#12-vercel-deployment)
13. [How to Extend This System](#13-how-to-extend-this-system)
14. [Designing Similar Systems](#14-designing-similar-systems-from-scratch)
15. [Common Mistakes & Best Practices](#15-common-mistakes--best-practices)
16. [Testing & Debugging](#16-testing--debugging)
17. [API Reference](#17-api-reference)
18. [Future Direction: Cloud Deployment](#18-future-direction-cloud-deployment)

---

## 1. What Is KG-MAG?

KG-MAG is a **Retrieval-Augmented Generation (RAG) system** that:

1. Accepts a corpus of documents (PDFs, Markdown, text files)
2. Builds a semantic search index over them
3. Uses a multi-agent AI pipeline to generate Medium-style articles
4. Cites every claim with source chunks
5. Runs automated quality checks (hallucination detection, readability, grounding)
6. Generates content-grounded technical diagrams inside the article body

The key property: **the system cannot hallucinate facts not present in your documents**. Every paragraph is verified against source material.

### Who is this for?

- AI/ML engineers building RAG systems for portfolios
- Researchers wanting to automatically summarize paper collections
- Teams generating internal knowledge base articles
- Anyone learning production RAG engineering

---

## 2. How the System Works

### Conceptual Overview

```
Your Documents
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  INGESTION PIPELINE                                 │
│  Extract → Clean → Chunk → Embed → Index           │
└─────────────────────────────────────────────────────┘
    │
    ▼  (FAISS Vector Index)
┌─────────────────────────────────────────────────────┐
│  MULTI-AGENT GENERATION PIPELINE                   │
│                                                     │
│  [Topic] → PlannerAgent → ArticleOutline           │
│               │                                     │
│               ▼                                     │
│          RetrieverAgent → Relevant Chunks           │
│               │                                     │
│               ▼                                     │
│          WriterAgent → Draft Article               │
│               │                                     │
│               ▼                                     │
│          CriticAgent → QA Report                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final Article + Images + Citations + QA Report
```

### The RAG Loop

The core insight of RAG: **instead of asking a model to recall facts from training data** (which leads to hallucination), you **retrieve relevant passages first**, then condition generation on them.

```
Query: "Explain transformer attention"
           │
           ▼
[Embed query] → vector
           │
           ▼
[FAISS search] → top-8 most similar chunks from YOUR documents
           │
           ▼
[LLM prompt] = "Using ONLY these passages: [chunks]... write about transformers"
           │
           ▼
Grounded output, citable back to your corpus
```

---

## 3. Quick Start

### Prerequisites

- Docker & Docker Compose
- API keys: OpenAI + Nanobananpro

### Step 1: Clone and configure

```bash
git clone https://github.com/SuyashPatil98/KG-MAG.git
cd KG-MAG
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
NANOBANANPRO_API_KEY=your_key_here
BACKEND_API_KEY=replace_with_a_long_random_secret
```

`BACKEND_API_KEY` is required when running in production mode.

### Step 2: Start everything

```bash
docker compose up --build
```

This will:

- Pull dependencies and download the embedding model (~90MB, runs locally)
- Start the FastAPI backend on port 8080 (host) / 8000 (container)
- Start the Next.js frontend on port 3000

Wait for the log line: `Application startup complete`.

### Security Defaults

- Production requires `BACKEND_API_KEY` (all `/api/*` endpoints enforce bearer auth).
- Browser requests are proxied through Next.js route handlers, so the backend key stays server-side.
- Abuse controls are enabled by default:
  - `RATE_LIMIT_GENERATE_REQUESTS=6` per `RATE_LIMIT_WINDOW_SECONDS=60`
  - `RATE_LIMIT_INGEST_REQUESTS=8` per `RATE_LIMIT_WINDOW_SECONDS=60`
  - `MAX_UPLOAD_FILES_PER_REQUEST=10`
  - `MAX_UPLOAD_FILE_SIZE_MB=20`

### Step 3: Add your documents

Open http://localhost:3000 and drag-drop your PDFs/Markdown/text files.
Or use the CLI:

```bash
python scripts/ingest_kb.py --dir ./my_documents
```

### Step 4: Generate an article

In the UI, enter a topic related to your documents and click **Generate Article**.

Or via API:

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Authorization: Bearer <your_backend_api_key>" \
  -H "Content-Type: application/json" \
  -d '{"topic": "The evolution of attention mechanisms in NLP"}'
```

---

## 4. Architecture Deep Dive

### Project Structure

```
kg-mag/
├── backend/
│   ├── agents/
│   │   └── orchestrator.py     # Multi-agent pipeline: Planner, Retriever, Writer, Critic
│   ├── api/
│   │   └── main.py             # FastAPI application, all routes
│   ├── core/
│   │   ├── config.py           # Pydantic-settings config (env-driven)
│   │   ├── logging.py          # Structlog structured logging
│   │   └── models.py           # Core Pydantic data models
│   ├── models/
│   │   └── reranker.py         # ML retrieval reranker (scikit-learn)
│   ├── tools/
│   │   ├── embedding.py        # Sentence-transformers wrapper
│   │   ├── vector_store.py     # FAISS persistent vector store
│   │   ├── llm_client.py       # OpenAI API wrapper
│   │   └── image_gen.py        # Nanobananpro image generation
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/                # Next.js App Router pages
│       ├── components/         # React components (ArticlePreview, QAReport, etc.)
│       └── lib/                # API client, TypeScript types
├── ingestion/
│   └── pipeline.py             # PDF/MD/TXT extraction, cleaning, chunking
├── configs/
│   └── nginx.conf              # Production reverse proxy
├── docker/
│   ├── Dockerfile.backend      # Multi-stage Python image
│   └── Dockerfile.frontend     # Multi-stage Node image
├── tests/
│   └── test_suite.py           # Comprehensive pytest suite
├── scripts/
│   └── ingest_kb.py            # CLI utilities
├── docker-compose.yml
├── .env.example
└── README.md                   # This file
```

### Technology Choices and Why

| Component  | Technology            | Why                                             |
| ---------- | --------------------- | ----------------------------------------------- |
| LLM        | OpenAI GPT models     | Strong technical writing and structured outputs |
| Embeddings | sentence-transformers | Free, runs locally, high quality                |
| Vector DB  | FAISS                 | No external service needed, deterministic       |
| Backend    | FastAPI               | Async, auto-docs, typed, production-ready       |
| Frontend   | Next.js 15            | App Router, Vercel-native, type-safe            |
| ML         | scikit-learn GBM      | Interpretable, fast, no GPU needed              |

---

## 5. Data Ingestion Pipeline

### How Chunking Works

Chunking is the most underrated decision in RAG. Too large → retrieval returns irrelevant context. Too small → retrieved chunks lack meaning.

**KG-MAG uses heading-aware semantic chunking:**

```
Document
    │
    ├── Split on headings (# ## ###)    ← Section boundaries
    │       │
    │       └── Sliding window within sections
    │               chunk_size = 512 words
    │               overlap    = 64 words
    │
    └── Each chunk inherits section heading as metadata
```

**Why overlapping chunks?**

Consider a key fact that appears at the boundary between two windows:

```
... end of window 1 | START OF WINDOW 2 ...
                    ^ fact lives here
```

Without overlap, a query retrieving window 1 misses the fact. With 64-word overlap, both windows contain it.

**Why heading-aware?**

Headings are natural semantic boundaries. An LLM asking "what is the conclusion?" should retrieve conclusion chunks, not introduction chunks. By resetting chunk indices at heading boundaries, we ensure retrieved chunks are contextually cohesive.

### Text Cleaning

Every text goes through normalization before chunking:

1. **Unicode NFC normalization** — fixes encoding artifacts from PDF extraction
2. **Null byte removal** — PDFs often contain control characters
3. **Ligature expansion** — `ﬁ` → `fi` (common in academic PDFs)
4. **Whitespace normalization** — preserves paragraph structure, collapses excess

### Metadata Extraction

Each chunk carries:

- `source_id`: SHA-256 hash of filename+content (enables deduplication)
- `filename`: original document name (for citations)
- `heading`: section heading if detected
- `chunk_index`: position in document (used as a position signal in reranker)
- `token_count`: used for context length budgeting in generation

---

## 6. Embedding & Retrieval

### The Embedding Model

We use `sentence-transformers/all-MiniLM-L6-v2`:

- 384-dimensional dense vectors
- 14,500 tokens/second on CPU
- Trained specifically for semantic similarity
- Downloads once, runs entirely offline

**Why not OpenAI embeddings?**

- Cost: $0.0001/1K tokens adds up for large corpora
- Latency: network round-trip vs. in-process inference
- Privacy: your documents never leave your server
- Consistency: local model won't change between API versions

### FAISS Index

FAISS (Facebook AI Similarity Search) stores and retrieves vectors:

```
IndexFlatIP wrapped in IndexIDMap2

IndexFlatIP = exact inner-product search
            = cosine similarity when vectors are L2-normalized (we normalize)

IndexIDMap2 = maps FAISS integer IDs ↔ our string chunk_ids
```

For corpora under 1M chunks, `IndexFlatIP` provides exact search with excellent performance. For larger corpora, swap for `IndexIVFFlat` (approximate, 10x faster).

### Retrieval Pipeline

```
Query string
    │
    ├─[1] Embed query → 384-dim vector
    │
    ├─[2] FAISS.search(vector, top_k=8)
    │       Returns 8 candidates with cosine similarity scores
    │
    ├─[3] ML Reranker
    │       6-feature model rescores candidates
    │       Returns top_k=4 best matches
    │
    └─[4] Return RetrievedChunk objects with scores
```

---

## 7. The Multi-Agent System

### Why Multi-Agent?

A single "write me an article" prompt produces mediocre output. Decomposing into specialized agents with distinct responsibilities yields measurably better results:

| Agent         | Single Responsibility | Input → Output            |
| ------------- | --------------------- | ------------------------- |
| **Planner**   | Strategic structure   | Topic → Outline           |
| **Retriever** | Information gathering | Sections → Chunks         |
| **Writer**    | Prose generation      | Outline+Chunks → Sections |
| **Critic**    | Quality verification  | Article → QA Report       |

This mirrors how professional publications actually work: editor, researcher, writer, fact-checker.

### MCP-Style Orchestration

Each agent:

1. Receives a shared `PipelineContext` object
2. Performs its single responsibility
3. Mutates and returns the context
4. Errors are logged but don't crash the pipeline

The `ArticleOrchestrator` chains them:

```python
ctx = PlannerAgent.run(ctx)      # Adds: ctx.outline
ctx = RetrieverAgent.run(ctx)    # Adds: ctx.retrieved
ctx = WriterAgent.run(ctx)       # Adds: ctx.article
ctx = CriticAgent.run(ctx)       # Adds: ctx.qa_report
```

### Writer Agent: Grounded Generation

The Writer Agent's system prompt includes a critical constraint:

> "ONLY write content that is directly supported by the provided source chunks. Insert citation markers [CITE:chunk_id] immediately after each factual claim."

This produces output like:

```
Transformer models use self-attention to weigh input tokens [abc123].
This allows parallel processing unlike RNNs [def456].
```

Later, the Critic Agent verifies these claims against the source material.

---

## 8. ML Reranking Model

### The Problem Reranking Solves

Embedding similarity captures semantic relatedness, but it's a blunt instrument. Consider:

- Query: "advantages of FAISS over Chroma"
- Chunk A (score: 0.72): "FAISS is 3x faster for datasets under 1M vectors"
- Chunk B (score: 0.70): "Chroma provides native metadata filtering"

Both are semantically similar. But Chunk A is more _relevant_ because it directly compares the two systems. A reranker, with richer features, can identify this.

### Features (6 total)

| #   | Feature           | Description                             |
| --- | ----------------- | --------------------------------------- | ------------ | ------- | --- | --- | --- | --- |
| 0   | `cosine_sim`      | FAISS inner-product score               |
| 1   | `keyword_overlap` | BM25-style term overlap: `              | intersection | / sqrt( | q   |     | c   | )`  |
| 2   | `query_coverage`  | Fraction of query terms found in chunk  |
| 3   | `heading_match`   | Topic alignment with section heading    |
| 4   | `length_score`    | Normalized chunk length                 |
| 5   | `position_score`  | Earlier in document = slight preference |

### Training

For portfolio use, we simulate training data:

- **Positive examples**: query derived from first 7 words of a chunk (the chunk is assumed relevant)
- **Negative examples**: same query, random unrelated chunks

In production, replace with:

- User click data (implicit feedback)
- Human relevance judgments (RLHF-style)
- Distilled labels from a large cross-encoder (e.g., `ms-marco-MiniLM`)

The model is a `GradientBoostingClassifier` — interpretable, fast, no GPU needed.

---

## 9. QA & Hallucination Detection

### The Problem

LLMs hallucinate. Even with RAG, a model can:

- Extrapolate beyond what the source says
- Confuse details between chunks
- Generate plausible-sounding but unsourced claims

### KG-MAG's 4-Layer Defense

**Layer 1: Grounding Verification**
Default mode is token-efficient heuristic grounding (`QA_GROUNDING_MODE=heuristic`):
paragraphs are treated as grounded when they include valid inline citations mapped to retrieved chunks.
Optional strict mode (`QA_GROUNDING_MODE=llm`) uses model-based grounding checks.

**Layer 2: Self-Consistency Check**
Token-free consistency uses local embeddings to measure semantic continuity between adjacent sections,
then blends citation density as a grounded coherence signal.

**Layer 3: Flesch Readability Score**
Measures reading ease (0–100). Academic/dense text scores low. We target ≥50 (standard reading level).

Formula:

```
206.835 - (1.015 × avg_words/sentence) - (84.6 × avg_syllables/word)
```

**Layer 4: Coverage Score**
What fraction of retrieved chunks are actually cited in the article?
Low coverage = the writer may have drifted from the source material.

### QA Report Example

```json
{
  "grounding_score": 0.87,
  "readability_score": 62.4,
  "coverage_score": 0.75,
  "consistency_score": 0.81,
  "overall_confidence": 0.79,
  "passed": true,
  "warnings": []
}
```

---

## 10. Image Generation

The `ImageGenerationTool` generates publication-style technical visuals and supports:

- Nanobananpro-style image APIs
- Google Generative Language `generateContent` image responses (`inlineData`)

Current default behavior is intentionally cost-aware and article-grounded:

- Generate one primary diagram at 1024x576 (16:9, 1k)
- Ground the prompt using section headings plus distilled retrieval evidence
- Prioritize relevance to article mechanisms over generic visuals
- Attach the generated image to the first article section (not page header)
- Keep image generation optional so article generation never blocks on image failures

Frontend rendering behavior:

- Displays the in-article diagram in a 16:9 container (`aspect-video`) with full visibility (`object-contain`)
- Provides a per-image "Download image" action in the article preview
- Includes section images in markdown export

---

## 11. Docker Deployment

### One Command

```bash
# Build and start all services
docker compose up --build

# Background mode
docker compose up --build -d

# View logs
docker compose logs -f backend

# Stop
docker compose down

# With Chroma instead of FAISS
docker compose --profile chroma up --build

# Production mode with Nginx
docker compose --profile production up --build
```

### Data Persistence

All data is stored in the `kg_data` Docker volume:

```
kg_data/
├── kb/               # FAISS index
├── uploads/          # Ingested documents
├── artifacts/        # Generated images
└── models/           # Trained reranker
```

To backup:

```bash
docker run --rm -v kg-mag_kg_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/kg-mag-backup.tar.gz /data
```

---

## 12. Vercel Deployment

### Step 1: Deploy the Backend

The FastAPI backend needs a server (Vercel doesn't support long-running Python processes well).
Use one of:

- **Railway**: `railway up` from the project root
- **Render**: Connect GitHub repo, set `docker/Dockerfile.backend`
- **AWS ECS**: Use the provided Dockerfile
- **Fly.io**: `fly launch` with the backend Dockerfile

After deployment, note your backend URL: `https://kg-mag-backend.railway.app`

### Step 2: Set Vercel Environment Variables

```bash
# Install Vercel CLI
npm i -g vercel

# Create server-side secrets
vercel env add BACKEND_INTERNAL_URL production
# Enter: https://kg-mag-backend.railway.app

vercel env add BACKEND_API_KEY production
# Enter: your_backend_api_key
```

Or set via the Vercel dashboard under `Project → Settings → Environment Variables`.

### Step 3: Deploy Frontend

```bash
cd frontend
vercel --prod
```

Vercel auto-detects Next.js and configures the build pipeline.

### Step 4: Update CORS

Add your Vercel URL to the backend's `CORS_ORIGINS` env var:

```env
CORS_ORIGINS=http://localhost:3000,https://kg-mag.vercel.app
```

### Environment Variables Reference

| Variable               | Where              | Description                                                 |
| ---------------------- | ------------------ | ----------------------------------------------------------- |
| `BACKEND_INTERNAL_URL` | Frontend server    | Backend base URL used by Next.js proxy route handlers       |
| `BACKEND_API_KEY`      | Frontend + Backend | Shared backend bearer key (never expose as `NEXT_PUBLIC_*`) |
| `OPENAI_API_KEY`       | Backend            | OpenAI API key                                              |
| `NANOBANANPRO_API_KEY` | Backend            | Image generation key                                        |

---

## 13. How to Extend This System

### Add New Document Types

In `ingestion/pipeline.py`, add an extractor:

```python
def extract_html(file_path: Path) -> tuple[str, DocumentMetadata]:
    from bs4 import BeautifulSoup
    raw = file_path.read_text()
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator="\n")
    cleaned = clean_text(text)
    meta = DocumentMetadata(filename=file_path.name, source_type=SourceType.HTML, ...)
    return cleaned, meta

# Register it:
EXTRACTORS[".html"] = extract_html
EXTRACTORS[".htm"] = extract_html
```

### Add a New Agent

Create a new agent class:

```python
class SEOAgent:
    """Optimizes the article for search engines post-generation."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.article
        # Add meta description, optimize headings, internal links...
        ctx.article.seo_keywords = self._extract_keywords(ctx.article)
        return ctx
```

Register it in the orchestrator:

```python
ctx = self._seo_agent.run(ctx)   # After WriterAgent
```

### Swap Vector DB to Chroma

Set `VECTOR_DB=chroma` in your `.env`. The system routes to ChromaDB automatically.
To implement: create `backend/tools/chroma_store.py` with the same interface as `FAISSVectorStore`.

### Add Streaming Generation

In `WriterAgent._write_section`, replace `llm.complete()` with `llm.stream()` and use Server-Sent Events (SSE) in FastAPI:

```python
from fastapi.responses import StreamingResponse

@app.post("/api/generate/stream")
async def generate_stream(request: GenerateRequest):
    async def event_generator():
        for token in llm.stream(system, user):
            yield f"data: {token}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### Add Multi-Language Support

Add a `language` field to `GenerateRequest` and inject it into agent prompts:

```python
system = f"You are a writer. Respond in {request.language}."
```

---

## 14. Designing Similar Systems from Scratch

### The Core RAG Design Loop

When building any RAG system, answer these questions in order:

**1. What is the retrieval unit?**

- Too small (sentence): fast but lacks context
- Too large (full page): slow and noisy
- Sweet spot: ~300–500 words with 10–15% overlap

**2. What is the embedding strategy?**

- Symmetric (same model for query and document): good for QA
- Asymmetric: specialized query encoder (SPLADE, ColBERT) for complex retrieval
- Start with `all-MiniLM-L6-v2`, upgrade to `bge-large-en` if recall is low

**3. What is the retrieval strategy?**

- Dense only (embeddings): good for semantic similarity
- Sparse only (BM25): good for keyword matching
- Hybrid (dense + sparse + reranker): best for production

**4. What is the grounding strategy?**

- Citation markers in generation (our approach)
- Post-hoc attribution (match sentences to chunks after generation)
- Constrained decoding (force model to only use source vocabulary)

**5. How do you measure quality?**

- MRR@10 for retrieval
- Grounding score for faithfulness
- ROUGE/BERTScore for text quality
- Human eval for readability

### The RAG Quality Hierarchy

```
Level 1: Can the system retrieve relevant chunks? (MRR, nDCG)
Level 2: Can the LLM use retrieved context? (Faithfulness score)
Level 3: Is the output readable and correct? (Readability + human eval)
Level 4: Does the output serve the user's actual need? (Task completion)
```

Most RAG projects fail at Level 1. Fix retrieval before fixing generation.

---

## 15. Common Mistakes & Best Practices

### Chunking Mistakes

❌ **Fixed-size chunking by character count**
Cuts words in half. Use word-count-based chunking.

❌ **Zero overlap**
Information at chunk boundaries is lost. Always use ≥10% overlap.

❌ **Ignoring document structure**
PDFs have headers, tables, figures. Don't treat them as plain text.

✅ **Heading-aware + overlapping** (what KG-MAG does)

### Retrieval Mistakes

❌ **Only using top-1 retrieval**
One bad retrieval tanks your generation. Use top-5 to top-10, rerank.

❌ **Not normalizing embeddings**
If you use cosine similarity, normalize vectors to L2=1 before indexing.

❌ **Treating retrieval as solved after the first prototype**
Retrieval is the hardest part of RAG. Measure MRR continuously.

✅ **Dense retrieval + BM25 hybrid + reranker**

### Generation Mistakes

❌ **Putting all retrieved chunks in one big context**
LLMs lose focus with too much context ("lost in the middle" problem).
Use reranking and limit to top-4 chunks per section.

❌ **Not constraining generation to source material**
Without explicit grounding instructions, LLMs mix retrieved facts with hallucinated ones.

❌ **One-shot generation without review**
Always have a critic pass. Even a simple readability check catches poor output.

✅ **Multi-agent: Plan → Retrieve → Write → Critique**

### Infrastructure Mistakes

❌ **Hardcoding API keys**
Use environment variables. Always. See `.env.example`.

❌ **Storing secrets in Docker images**
Use build args for non-sensitive config only. Runtime secrets via env vars.

❌ **No health checks**
FastAPI + Docker Compose health checks prevent silent failures.

✅ **12-factor app principles** (config via environment, stateless processes)

---

## 16. Testing & Debugging

### Run the Test Suite

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_suite.py::TestVectorStore -v

# With coverage
pytest tests/ --cov=backend --cov=ingestion --cov-report=html

# Fast mode (skip slow embedding tests)
pytest tests/ -v -k "not TestEmbedding"
```

### Debug Ingestion

```bash
# Validate your documents can be ingested
python scripts/ingest_kb.py --dir ./my_docs

# Check what was indexed
python scripts/ingest_kb.py --stats

# Test retrieval quality
python scripts/ingest_kb.py --validate

# Test a specific query
python scripts/ingest_kb.py --query "your test query here" --top-k 5
```

### Debug Agent Failures

1. Check the pipeline log in the UI (every step is logged)
2. Check backend logs: `docker compose logs backend`
3. Enable debug logging: `LOG_LEVEL=DEBUG` in `.env`
4. Inspect individual agent outputs by adding print statements to `orchestrator.py`

### Debug Retrieval Quality

If the article seems off-topic:

```bash
python scripts/ingest_kb.py --validate
```

Low MRR (< 0.4)?

- Documents may not cover the topic well
- Try adding more relevant source material
- Adjust `CHUNK_SIZE` (smaller = more precise retrieval)

### Debug Hallucinations

If the QA report shows low grounding score:

1. Check if your documents actually contain information about the topic
2. Lower `QA_GROUNDING_THRESHOLD` if the topic is inherently abstract
3. Add more source material related to the topic
4. Reduce `MAX_ARTICLE_TOKENS` to force shorter, more focused sections

---

## 17. API Reference

When running via Docker Compose, use backend base URL: `http://localhost:8080`.

### POST `/api/ingest`

Upload documents to the knowledge base.

**Form data:** `files` (multipart, supports `.pdf`, `.md`, `.txt`)

**Response:**

```json
{
  "job_id": "uuid",
  "status": "completed",
  "chunks_created": 142,
  "documents_processed": 3,
  "duration_seconds": 4.2
}
```

### POST `/api/generate`

Generate a grounded article.

**Body:**

```json
{
  "topic": "The impact of BERT on NLP benchmarks",
  "target_audience": "ML engineers",
  "tone": "technical and precise",
  "generate_images": true,
  "run_qa": true,
  "max_sections": 5
}
```

**Response:** Full `GenerateResponse` with article and QA report.

### GET `/api/kb/status`

Returns knowledge base statistics.

### GET `/api/uploads`

Lists uploaded files with size, upload time, chunk count, and indexing status.

### POST `/api/uploads/delete`

Deletes selected uploaded files and rebuilds the index from remaining uploads.

### POST `/api/kb/rebuild`

Rebuilds vector index from uploaded documents.

### POST `/api/kb/reset`

Resets corpus state, with options to delete uploads and generated artifacts.

### GET `/api/dashboard/metrics`

Returns aggregate generation telemetry including duration, token usage, image counts, and QA pass/fail metrics.

### GET `/api/dashboard/logs`

Returns recent generation runs with stage timings and QA/image summaries.

### GET `/api/article/{article_id}`

Fetches a previously generated article by ID.

### GET `/api/articles`

Lists generated article IDs and titles.

### DELETE `/api/kb/clear`

Clears vector index and in-memory generated artifacts metadata.

### GET `/health`

Health check. Returns `200` when ready.

---

## 18. Future Direction: Cloud Deployment

The current project already supports local Docker deployment and GHCR image publishing. The next production milestone is a cloud-native deployment path.

### Target Cloud Architecture

1. Deploy backend and frontend containers to a managed runtime (AWS ECS/Fargate, GCP Cloud Run, Azure Container Apps, or Railway/Render).
2. Move local `/data` paths to managed persistent storage and backups.
3. Put services behind managed HTTPS ingress with secrets in a secret manager.
4. Centralize logs and add metrics/alerts for health and latency.

### Recommended Implementation Phases

1. **Phase 1: Lift-and-shift containers**
   - Reuse current Docker images from GHCR with environment-specific configuration.
2. **Phase 2: Managed persistence**
   - Externalize document/artifact storage and define retention/backup policies.
3. **Phase 3: Production hardening**
   - Add autoscaling, rate limiting, and staged deployments (`staging` to `prod`).
4. **Phase 4: Platform reliability**
   - Add tracing, SLO dashboards, and incident runbooks.

### Why This Matters

- Lowers onboarding friction for contributors and demo users.
- Makes runtime behavior reproducible beyond local environments.
- Prepares the codebase for multi-user, production-grade workloads.

---

## Contributing

1. Fork the repository and clone your fork.
2. Create a focused branch from `main`:

- `git checkout -b feature/your-change`

3. Make your changes and add tests where behavior changes.
4. Run checks before pushing:

- Backend: `pytest tests/ -v`
- Frontend: `cd frontend && npm ci && npm run lint && npm run build`

5. Push branch and open a pull request with a clear summary and test evidence.
6. Follow the full guide in [CONTRIBUTING.md](CONTRIBUTING.md).

### Run Using Published GHCR Images

Use this to validate the exact published containers (no local build required):

```bash
docker login ghcr.io
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up -d --no-build
```

Verify services:

- Frontend: http://localhost:3000
- Backend health: http://localhost:8080/health

## License

MIT License — see [LICENSE](LICENSE)

---

_Built as a portfolio-quality demonstration of production RAG engineering. Every design decision is documented above. Feel free to use this as a reference or starting point._

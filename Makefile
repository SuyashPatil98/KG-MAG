# KG-MAG — Developer Makefile
# Usage: make <target>

.PHONY: help dev build up down logs test ingest clean

DOCKER_COMPOSE = docker compose
PYTHON = python3

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────────

build:  ## Build all Docker images
	$(DOCKER_COMPOSE) build

up:  ## Start all services
	$(DOCKER_COMPOSE) up -d

up-dev:  ## Start with logs streaming
	$(DOCKER_COMPOSE) up

down:  ## Stop all services
	$(DOCKER_COMPOSE) down

logs:  ## Stream backend logs
	$(DOCKER_COMPOSE) logs -f backend

restart-backend:  ## Rebuild and restart backend only
	$(DOCKER_COMPOSE) up -d --build backend

shell-backend:  ## Open shell in running backend container
	$(DOCKER_COMPOSE) exec backend /bin/bash

# ── Development (local, no Docker) ────────────────────────────────────────────

install:  ## Install Python dependencies locally
	pip install -r backend/requirements.txt

install-dev:  ## Install with dev extras
	pip install -r backend/requirements.txt pytest pytest-asyncio pytest-cov

backend-dev:  ## Run backend in dev mode (hot reload)
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

frontend-dev:  ## Run Next.js dev server
	cd frontend && npm run dev

# ── Knowledge Base ────────────────────────────────────────────────────────────

ingest:  ## Ingest documents from ./data/uploads
	$(PYTHON) scripts/ingest_kb.py --dir ./data/uploads

ingest-dir:  ## Ingest from custom dir: make ingest-dir DIR=./my_docs
	$(PYTHON) scripts/ingest_kb.py --dir $(DIR)

kb-stats:  ## Print knowledge base statistics
	$(PYTHON) scripts/ingest_kb.py --stats

kb-validate:  ## Validate retrieval quality
	$(PYTHON) scripts/ingest_kb.py --validate

kb-clear:  ## Clear the knowledge base (DESTRUCTIVE)
	$(PYTHON) scripts/ingest_kb.py --dir /dev/null --rebuild

train-reranker:  ## Train the ML reranker model
	$(PYTHON) scripts/ingest_kb.py --train-reranker

# ── Testing ───────────────────────────────────────────────────────────────────

test:  ## Run full test suite
	pytest tests/ -v

test-fast:  ## Run tests excluding slow embedding tests
	pytest tests/ -v -k "not Embedding"

test-cov:  ## Run tests with coverage report
	pytest tests/ --cov=backend --cov=ingestion --cov-report=html --cov-report=term-missing

test-ingestion:  ## Test ingestion pipeline only
	pytest tests/test_suite.py::TestIngestion -v

test-retrieval:  ## Test vector store only
	pytest tests/test_suite.py::TestVectorStore -v

test-qa:  ## Test QA system only
	pytest tests/test_suite.py::TestQA -v

# ── Utilities ─────────────────────────────────────────────────────────────────

env-check:  ## Verify .env is configured
	@python3 -c "from backend.core.config import get_settings; s = get_settings(); print('✓ Config OK'); print(f'  LLM model: {s.llm_model}'); print(f'  Vector DB: {s.vector_db}'); print(f'  Embedding: {s.embedding_model}')"

lint:  ## Run ruff linter
	ruff check backend/ ingestion/ tests/

format:  ## Format code with ruff
	ruff format backend/ ingestion/ tests/

type-check:  ## Run mypy type checker
	mypy backend/ ingestion/ --ignore-missing-imports

clean:  ## Remove __pycache__, .pytest_cache, build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "✓ Cleaned"

# ── Backup ────────────────────────────────────────────────────────────────────

backup-kb:  ## Backup knowledge base to ./backups/
	mkdir -p ./backups
	tar czf ./backups/kg-data-$(shell date +%Y%m%d-%H%M%S).tar.gz ./data/
	@echo "✓ Knowledge base backed up to ./backups/"

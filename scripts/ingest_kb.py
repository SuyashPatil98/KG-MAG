#!/usr/bin/env python3
"""
KG-MAG — CLI Scripts
======================
Collection of utility scripts for managing the knowledge base.

Usage
-----
# Ingest a directory of documents
python scripts/ingest_kb.py --dir ./my_docs

# Test retrieval quality
python scripts/ingest_kb.py --query "machine learning applications"

# Train the reranker
python scripts/ingest_kb.py --train-reranker

# Print KB stats
python scripts/ingest_kb.py --stats
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_ingest(directory: str, rebuild: bool = False) -> None:
    from ingestion.pipeline import ingest_directory
    from backend.tools.vector_store import FAISSVectorStore
    from backend.core.logging import setup_logging, get_logger

    setup_logging("INFO")
    log = get_logger("ingest_cli")

    store = FAISSVectorStore()

    if rebuild:
        log.info("Clearing existing index...")
        store.clear()

    doc_dir = Path(directory)
    if not doc_dir.exists():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    t0 = time.perf_counter()
    total_chunks = 0
    doc_count = 0

    print(f"\n{'─'*60}")
    print("  KG-MAG Ingestion Pipeline")
    print(f"  Source: {doc_dir.resolve()}")
    print(f"{'─'*60}\n")

    for chunks in ingest_directory(doc_dir):
        if not chunks:
            continue
        filename = chunks[0].filename
        added = store.add_chunks(chunks)
        total_chunks += added
        doc_count += 1
        print(f"  ✓ {filename:<40} → {added:>4} chunks")

    store.save()
    elapsed = time.perf_counter() - t0

    print(f"\n{'─'*60}")
    print(f"  Documents: {doc_count}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Duration:  {elapsed:.2f}s")
    print(f"  Index:     {store.total_vectors} vectors")
    print(f"{'─'*60}\n")


def cmd_query(query: str, top_k: int = 5) -> None:
    from backend.tools.vector_store import FAISSVectorStore

    store = FAISSVectorStore()
    if store.total_vectors == 0:
        print("ERROR: Knowledge base is empty. Run ingest first.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Query: '{query}'")
    print(f"  Top-K: {top_k}")
    print(f"{'─'*60}\n")

    results = store.search(query, top_k=top_k)

    for r in results:
        heading = f"[{r.chunk.heading}] " if r.chunk.heading else ""
        print(f"  Rank #{r.rank}  Score: {r.score:.4f}  {heading}")
        print(f"  File: {r.chunk.filename} | Chunk #{r.chunk.chunk_index}")
        print(f"  Text: {r.chunk.text[:200]}...")
        print()


def cmd_stats() -> None:
    from backend.tools.vector_store import FAISSVectorStore
    from backend.core.config import get_settings

    cfg = get_settings()
    store = FAISSVectorStore()
    chunks = store.all_chunks()

    sources = {}
    for c in chunks:
        sources.setdefault(c.filename, []).append(c)

    print(f"\n{'─'*60}")
    print("  KG-MAG Knowledge Base Statistics")
    print(f"{'─'*60}")
    print(f"  Vector DB:       {cfg.vector_db.upper()}")
    print(f"  Embedding Model: {cfg.embedding_model}")
    print(f"  Total Vectors:   {store.total_vectors}")
    print(f"  Total Documents: {len(sources)}")
    print("\n  Documents:")
    for fname, doc_chunks in sorted(sources.items()):
        avg_len = sum(len(c.text.split()) for c in doc_chunks) // len(doc_chunks)
        print(f"    {fname:<40} {len(doc_chunks):>4} chunks  ~{avg_len} words/chunk")
    print(f"{'─'*60}\n")


def cmd_train_reranker() -> None:
    from backend.tools.vector_store import FAISSVectorStore
    from backend.models.reranker import train_reranker

    store = FAISSVectorStore()
    chunks = store.all_chunks()

    if len(chunks) < 10:
        print(f"ERROR: Need at least 10 chunks to train. Found {len(chunks)}.")
        sys.exit(1)

    print(f"\nTraining reranker on {len(chunks)} chunks...")
    t0 = time.perf_counter()
    train_reranker(chunks)
    elapsed = time.perf_counter() - t0
    print(f"✓ Reranker trained and saved in {elapsed:.2f}s\n")


def cmd_validate_retrieval() -> None:
    """Run a set of test queries and score retrieval quality."""
    from backend.tools.vector_store import FAISSVectorStore

    store = FAISSVectorStore()
    if store.total_vectors == 0:
        print("ERROR: Empty KB.")
        sys.exit(1)

    # Use actual chunk content as ground truth queries
    chunks = store.all_chunks()[:5]

    print(f"\n{'─'*60}")
    print("  Retrieval Quality Validation")
    print(f"{'─'*60}\n")

    total_mrr = 0.0
    for i, target_chunk in enumerate(chunks):
        # Use first 8 words as query
        words = target_chunk.text.split()
        query = " ".join(words[: min(8, len(words))])

        results = store.search(query, top_k=10)
        retrieved_ids = [r.chunk.chunk_id for r in results]

        # MRR: reciprocal rank of the ground-truth chunk
        try:
            rank = retrieved_ids.index(target_chunk.chunk_id) + 1
            mrr = 1.0 / rank
            label = "✓"
        except ValueError:
            rank = -1
            mrr = 0.0
            label = "✗"

        total_mrr += mrr
        print(f"  {label} Query: '{query[:50]}...'")
        print(f"    Target rank: {rank}  MRR contribution: {mrr:.3f}\n")

    mean_mrr = total_mrr / len(chunks)
    print(f"  Mean Reciprocal Rank (MRR@10): {mean_mrr:.4f}")
    print(f"  {'GOOD' if mean_mrr > 0.5 else 'NEEDS IMPROVEMENT'}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG-MAG CLI Utilities")
    parser.add_argument("--dir", help="Ingest documents from directory")
    parser.add_argument(
        "--rebuild", action="store_true", help="Clear existing index before ingesting"
    )
    parser.add_argument("--query", help="Test retrieval with a query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stats", action="store_true", help="Print KB statistics")
    parser.add_argument(
        "--train-reranker", action="store_true", help="Train reranker model"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate retrieval quality"
    )

    args = parser.parse_args()

    if args.dir:
        cmd_ingest(args.dir, rebuild=args.rebuild)
    elif args.query:
        cmd_query(args.query, top_k=args.top_k)
    elif args.stats:
        cmd_stats()
    elif args.train_reranker:
        cmd_train_reranker()
    elif args.validate:
        cmd_validate_retrieval()
    else:
        parser.print_help()

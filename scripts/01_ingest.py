"""Build the Chroma vector index from the pre-built corpora."""
"""Stage 1: Ingest corpus documents and build vector index.

This script orchestrates the ingestion pipeline:
1. Reads PDF documents from ORG_CORPORA (pre-extracted JSON files)
2. Reads press articles from ORG_ARTICLE_DIRS (RTF files)
3. Chunks documents into overlapping segments
4. Embeds chunks using the Together embedding API
5. Stores embeddings and metadata in Chroma vector database

Usage:
    python scripts/01_ingest.py             # Build index (resume if interrupted)
    python scripts/01_ingest.py --reset     # Delete existing index and rebuild
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.ingest import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Optional --reset flag: delete and rebuild the collection from scratch
    # Useful if you want to re-ingest after updating the corpus or configuration
    ap.add_argument("--reset", action="store_true", help="drop existing collection first")
    args = ap.parse_args()
    # Call the ingestion function with the reset flag
    build_index(reset=args.reset)

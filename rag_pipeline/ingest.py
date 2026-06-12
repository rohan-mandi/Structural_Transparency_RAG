"""Document ingestion and embedding pipeline.

This module implements a robust end-to-end pipeline for:
1. Loading corpus documents (PDFs as JSON) and press articles (RTF files)
2. Splitting documents into overlapping chunks for semantic search
3. Embedding chunks using the Together embedding API
4. Storing embeddings + metadata in Chroma vector database

Key features:
- Resilient to per-item embedding failures (bisects batch on 400 errors)
- Retry logic with exponential backoff for transient API errors
- Handles RTF articles that may contain multiple concatenated articles
- Skips chunks that are too large for the embedding model
"""
import json
import re
import time
from typing import Iterator

import chromadb
from chromadb.config import Settings
from striprtf.striprtf import rtf_to_text
from together import Together
from tqdm import tqdm

from .config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, ORG_CORPORA, ORG_ARTICLE_DIRS,
    TOGETHER_API_KEY, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS,
)

# Batch size for embedding calls (trade-off between API efficiency and memory)
EMBED_BATCH = 32

# Regex pattern to split RTF articles (many articles may be concatenated in one file)
ARTICLE_SPLIT_RE = re.compile(
    r"\n(?=(?:End of Document|Title:|Headline:|HEADLINE:|Section:|Byline:|Publication-Date:|Length:\s*\d+\s*words)\b)",
    re.IGNORECASE,
)


def chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks, respecting sentence boundaries.
    
    Uses a sliding window approach with intelligent break points:
    - Prefers breaking at newlines within the target size range
    - Falls back to period+space breaks if no newline found
    - Ensures minimum chunk size to avoid tiny fragments
    
    Args:
        text: Text to chunk
        size: Target chunk size in characters
        overlap: Number of overlapping characters between consecutive chunks
        
    Returns:
        list[str]: List of non-empty chunks
    """
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # If not at end, try to break at natural boundaries
        if end < len(text):
            # Prefer breaking at newlines
            window = text.rfind("\n", start, end)
            if window == -1 or window <= start + size // 2:
                # Fall back to periods
                window = text.rfind(". ", start, end)
            # Use boundary if found in reasonable position
            if window != -1 and window > start + size // 2:
                end = window + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        # Next chunk starts with overlap
        start = end - overlap
    # Filter out empty chunks
    return [c for c in chunks if c]


def _iter_pdf_corpus() -> Iterator[dict]:
    """Iterator: yield document chunks from pre-built PDF corpus.
    
    Loads JSON files containing pre-extracted PDFs, chunks each document,
    and yields chunks with full metadata (org, filename, chunk index, etc.).
    
    Yields:
        dict: Item with keys: 'id', 'text', 'metadata'
    """
    for org, path in ORG_CORPORA.items():
        if not path.exists():
            print(f"[skip] {org}: {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for doc in docs:
            text = doc.get("text", "")
            if not text.strip():
                continue
            # Chunk each document and yield with metadata
            for i, chunk in enumerate(chunk_text(text)):
                yield {
                    "id": f"{org}::pdf::{doc['filename']}::chunk{i}",
                    "text": chunk,
                    "metadata": {
                        "org": org,
                        "doc_type": "pdf",
                        "filename": doc.get("filename", ""),
                        "path": doc.get("path", ""),
                        "page_count": doc.get("page_count", 0),
                        "chunk_index": i,
                    },
                }


def _split_rtf_articles(text: str) -> list[str]:
    """Split concatenated RTF articles into individual articles.
    
    RTF files often bundle multiple press clippings. This function splits
    on common article separators (End of Document, Title, Headline, etc.).
    Falls back to the whole text if no meaningful split is found.
    
    Args:
        text: Plain text extracted from RTF file
        
    Returns:
        list[str]: List of individual articles
    """
    # Split on article boundaries
    parts = [p.strip() for p in ARTICLE_SPLIT_RE.split(text) if p and p.strip()]
    # If we got multiple parts, use them; otherwise return the whole text
    return parts if len(parts) > 1 else [text.strip()]


def _iter_rtf_articles() -> Iterator[dict]:
    """Iterator: yield document chunks from RTF article files.
    
    Converts RTF files to plain text, splits into individual articles,
    chunks each article, and yields with full metadata.
    
    Handles:
    - RTF parsing errors (skips problematic files)
    - Multiple concatenated articles in one file
    - Minimum article length filtering (200 chars)
    
    Yields:
        dict: Item with keys: 'id', 'text', 'metadata'
    """
    for org, dirpath in ORG_ARTICLE_DIRS.items():
        if not dirpath.exists():
            continue
        # Process both .RTF and .rtf files
        for rtf_path in sorted(dirpath.glob("*.RTF")) + sorted(dirpath.glob("*.rtf")):
            try:
                # Read and convert RTF to plain text
                with open(rtf_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                plain = rtf_to_text(raw, errors="ignore")
            except Exception as e:
                print(f"[skip] {rtf_path.name}: rtf parse error {e}")
                continue
            # Split into individual articles and chunk each one
            articles = _split_rtf_articles(plain)
            for art_idx, article in enumerate(articles):
                if len(article) < 200:  # Skip very short articles
                    continue
                for i, chunk in enumerate(chunk_text(article)):
                    yield {
                        "id": f"{org}::article::{rtf_path.stem}::a{art_idx}::chunk{i}",
                        "text": chunk,
                        "metadata": {
                            "org": org,
                            "doc_type": "article",
                            "filename": rtf_path.name,
                            "path": str(rtf_path),
                            "article_index": art_idx,
                            "chunk_index": i,
                        },
                    }


def iter_corpus() -> Iterator[dict]:
    """Iterator over all corpus items (PDFs + RTF articles).
    
    Yields:
        dict: Chunked documents from both PDF corpus and RTF articles
    """
    yield from _iter_pdf_corpus()
    yield from _iter_rtf_articles()


def _embed_call(client: Together, texts: list[str]) -> list[list[float]]:
    """Call the embedding API on a batch of texts.
    
    Args:
        client: Together API client
        texts: List of texts to embed
        
    Returns:
        list[list[float]]: List of embedding vectors
    """
    resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in resp.data]


def _is_oversize_error(msg: str) -> bool:
    """Check if error is due to input being too large for the embedder.
    
    Args:
        msg: Error message
        
    Returns:
        bool: True if this is an oversize/context-length error
    """
    m = msg.lower()
    return "400" in msg or "maximum context length" in m or "too long" in m


def _is_transient_error(msg: str) -> bool:
    """Check if error is transient/retryable (5xx, rate limit, connection).
    
    Args:
        msg: Error message
        
    Returns:
        bool: True if this is a transient error that should be retried
    """
    m = msg.lower()
    return any(s in msg for s in ("502", "503", "504", "429")) or "connection" in m or "timeout" in m


# Backoff schedule for retrying transient errors (in seconds)
TRANSIENT_BACKOFF = [2, 5, 10, 20, 40, 60, 60]


def embed_batch(client: Together, texts: list[str]) -> list[list[float] | None]:
    """Embed a batch of texts with resilient error handling.
    
    Handles two types of errors differently:
    - Transient errors (5xx, 429, connection): Retry with exponential backoff
      (~3 min total). If all retries fail, raise to alert caller that the
      service is unhealthy.
    - Oversize errors (400, context-length): Bisect the batch and recursively
      embed smaller chunks. Returns None for chunks that are too large to embed.
    
    This design ensures we don't silently drop chunks during API outages, but
    we do handle oversized individual chunks gracefully.
    
    Args:
        client: Together API client
        texts: Batch of texts to embed
        
    Returns:
        list[list[float] | None]: Embedding vectors, or None for oversized chunks
        
    Raises:
        RuntimeError: If transient errors persist after all retries
    """
    for attempt, wait in enumerate(TRANSIENT_BACKOFF):
        try:
            return _embed_call(client, texts)
        except Exception as e:
            msg = str(e)
            if _is_oversize_error(msg):
                return _bisect_or_drop(client, texts)
            if _is_transient_error(msg):
                print(f"  transient embed error (attempt {attempt+1}, sleep {wait}s): {e.__class__.__name__}")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(
        "Together embeddings API kept returning 5xx/connection errors after extended retry. "
        "The endpoint is likely down. Try `python scripts/01_ingest.py --reset` again later."
    )


def _bisect_or_drop(client: Together, texts: list[str]) -> list[list[float] | None]:
    """Handle oversized texts by bisecting batch or dropping single items.
    
    If a batch is too large for the API, try splitting it in half recursively.
    If a single text is too large, drop it and return None as its embedding.
    
    Args:
        client: Together API client
        texts: Batch of texts (at least one is oversized)
        
    Returns:
        list[list[float] | None]: Embeddings with None for dropped chunks
    """
    if len(texts) == 1:
        print(f"  [drop] chunk rejected as oversize by embedder (len={len(texts[0])} chars)")
        return [None]
    mid = len(texts) // 2
    return embed_batch(client, texts[:mid]) + embed_batch(client, texts[mid:])


def build_index(reset: bool = False) -> None:
    """Build the complete vector index from corpus documents.
    
    This is the main function that orchestrates the ingestion pipeline:
    1. Initialize Chroma collection (optionally reset if it exists)
    2. Iterate over all corpus documents (PDFs and RTF articles)
    3. Batch texts and send to embedding API
    4. Store embeddings + metadata in Chroma
    5. Print summary statistics
    
    The process is resumable: if the API fails partway, you can run it again
    and it will pick up where it left off (unless --reset flag is used).
    
    Args:
        reset: If True, delete any existing collection before rebuilding
        
    Raises:
        RuntimeError: If TOGETHER_API_KEY is not set
    """
    if not TOGETHER_API_KEY:
        raise RuntimeError("TOGETHER_API_KEY not set. Copy .env.example to .env and fill it in.")

    # Connect to Chroma vector database
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    # Optionally reset the collection
    if reset:
        try:
            chroma.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    # Get or create the collection (using cosine similarity for distance metric)
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Initialize Together API client
    client = Together(api_key=TOGETHER_API_KEY)
    # Buffers for batching items before embedding
    buf_ids, buf_texts, buf_meta = [], [], []
    total = 0  # Total successfully indexed chunks
    dropped = 0  # Chunks that were oversized and dropped

    def flush():
        """Embed buffered texts and upsert to Chroma."""
        nonlocal total, dropped
        if not buf_ids:
            return
        # Embed the batch
        embeddings = embed_batch(client, buf_texts)
        # Filter out None embeddings (oversized chunks) and prepare for upsert
        keep_ids, keep_embs, keep_docs, keep_meta = [], [], [], []
        for i, emb in enumerate(embeddings):
            if emb is None:
                dropped += 1
                continue
            keep_ids.append(buf_ids[i])
            keep_embs.append(emb)
            keep_docs.append(buf_texts[i])
            keep_meta.append(buf_meta[i])
        # Upsert to Chroma (insert or update if id already exists)
        if keep_ids:
            collection.upsert(ids=keep_ids, embeddings=keep_embs, documents=keep_docs, metadatas=keep_meta)
            total += len(keep_ids)
        # Clear buffers
        buf_ids.clear(); buf_texts.clear(); buf_meta.clear()

    # Iterate over corpus with progress bar
    pbar = tqdm(iter_corpus(), desc="ingest", unit="chunk")
    for item in pbar:
        buf_ids.append(item["id"])
        buf_texts.append(item["text"])
        buf_meta.append(item["metadata"])
        # Flush when buffer is full
        if len(buf_ids) >= EMBED_BATCH:
            flush()
            pbar.set_postfix(indexed=total, dropped=dropped)
    # Flush any remaining items
    flush()
    print(f"indexed {total} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR} (dropped {dropped})")

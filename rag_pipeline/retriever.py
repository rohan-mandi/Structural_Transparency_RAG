"""Chroma vector database wrapper for semantic search and retrieval.

This module provides a high-level interface to the Chroma vector database:
- Embedding queries using the same model as the corpus
- Semantic search to find the k most relevant chunks
- Optional filtering by organization
"""
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings
from together import Together

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOGETHER_API_KEY, TOP_K


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata.
    
    Attributes:
        id: Unique identifier for the chunk
        text: The actual text content
        org: Organization source (Anthropic, DeepMind, or OpenAI)
        filename: Original filename the chunk came from
        score: Similarity score to the query [0, 1]
    """
    id: str
    text: str
    org: str
    filename: str
    score: float


class Retriever:
    """Interface to Chroma vector database for semantic search.
    
    Handles:
    - Connection to persistent Chroma database
    - Query embedding using Together API
    - Semantic similarity search with optional org filtering
    """
    
    def __init__(self):
        """Initialize Chroma client and get the collection."""
        chroma = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        self.collection = chroma.get_collection(COLLECTION_NAME)
        self.client = Together(api_key=TOGETHER_API_KEY)

    def embed(self, text: str) -> list[float]:
        """Embed a text string using the Together embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            list[float]: Embedding vector of dimension EMBEDDING_DIM
        """
        resp = self.client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return resp.data[0].embedding

    def retrieve(self, query: str, k: int = TOP_K, org: Optional[str] = None) -> list[RetrievedChunk]:
        """Retrieve the k most semantically similar chunks to a query.
        
        Args:
            query: The search query text
            k: Number of results to return (default: TOP_K from config)
            org: Optional organization filter (Anthropic, DeepMind, OpenAI).
                 If provided, only retrieves chunks from that org.
                 
        Returns:
            list[RetrievedChunk]: The k most similar chunks, ranked by score
        """
        # Build filter for organization if specified
        where = {"org": org} if org else None
        # Embed the query
        emb = self.embed(query)
        # Query the Chroma collection using cosine similarity
        result = self.collection.query(
            query_embeddings=[emb], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        # Extract results from Chroma response
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]  # Distances from query
        # Convert distances to similarity scores (1 - distance)
        return [
            RetrievedChunk(
                id=ids[i], text=docs[i], org=metas[i].get("org", ""),
                filename=metas[i].get("filename", ""), score=1.0 - dists[i],
            )
            for i in range(len(ids))
        ]

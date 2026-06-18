"""RAG (Retrieval-Augmented Generation) question-answering pipeline.

Implements the standard RAG flow:
1. Retrieve: Find k most relevant document chunks using semantic search
2. Augment: Format chunks as context for the LLM
3. Generate: Ask the LLM to answer the question based on retrieved context

Key design note: The LLM sees the corpus context but NOT the institutional-logics
matrix. That matrix is reference knowledge that only the answer matcher uses.
"""
from dataclasses import dataclass
from typing import Optional

from .llm import chat
from .retriever import RetrievedChunk, Retriever

# System prompt for the LLM: defines its role and behavior
SYSTEM_PROMPT = (
    "You are an analyst of AI alignment governance. Answer the user's question "
    "using only the provided source excerpts. Be concise and specific. "
    "If the excerpts do not contain enough information, say so explicitly."
)

# Template for formatting the user message with question and context
USER_TEMPLATE = """Question:
{question}

Source excerpts:
{context}

Answer the question based on the excerpts above. State your conclusion first, then briefly justify it."""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a readable context string for the LLM.
    
    Each chunk is prefixed with its source metadata to help the LLM track where
    information came from.
    
    Args:
        chunks: List of retrieved chunks
        
    Returns:
        str: Formatted context with numbered sources
    """
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (org={c.org}, file={c.filename})\n{c.text}")
    return "\n\n".join(blocks)


@dataclass
class RAGAnswer:
    """Result of running the RAG pipeline.
    
    Attributes:
        question: The original question
        answer: The LLM's free-form answer based on retrieved context
        retrieved: List of document chunks that were used as context
    """
    question: str
    answer: str
    retrieved: list[RetrievedChunk]


def answer_question(retriever: Retriever, question: str, *, org: Optional[str] = None, k: int = 5) -> RAGAnswer:
    """Run the RAG pipeline: retrieve relevant chunks, then ask LLM to answer.
    
    Args:
        retriever: Retriever instance for semantic search
        question: The question to answer
        org: Optional organization filter (Anthropic, DeepMind, OpenAI).
             If provided, only retrieves from that organization's corpus.
        k: Number of chunks to retrieve (default 5)
        
    Returns:
        RAGAnswer: Question, LLM answer, and retrieved context
    """
    # Step 1: Retrieve most relevant chunks
    chunks = retriever.retrieve(question, k=k, org=org)
    # Step 2: Format retrieved chunks as context
    user = USER_TEMPLATE.format(question=question, context=_format_context(chunks))
    # Step 3: Ask LLM to answer based on context
    response = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ], temperature=0.0, max_tokens=512)
    return RAGAnswer(question=question, answer=response.strip(), retrieved=chunks)


CLOSED_BOOK_SYSTEM = (
    "You are an expert in organizational theory and institutional logics. "
    "Answer the question directly and concisely from your own knowledge. "
    "Commit to a single answer; do not hedge across alternatives."
)


def answer_closed_book(question: str) -> RAGAnswer:
    """Answer without retrieval — used for the theory eval set, whose questions
    are about institutional logics theory rather than the indexed corpus."""
    response = chat([
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": question},
    ], temperature=0.0, max_tokens=512)
    return RAGAnswer(question=question, answer=response.strip(), retrieved=[])

"""
RAG Workflow Tutorial - Package initialization file.

This package contains the core components of a Retrieval-Augmented Generation (RAG) system
for analyzing AI alignment governance through the lens of institutional logics theory.

The workflow consists of three main stages:
1. Ingest: Load documents, chunk text, embed with neural models, store in vector DB
2. Generate Eval: Create QA pairs from corpus chunks using LLM + institutional logic matrix
3. Run Eval: Execute RAG pipeline and grade answers using an answer matcher
"""

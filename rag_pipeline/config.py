"""Configuration module: paths, API keys, and model/pipeline hyperparameters.

All settings are defined here to make the system easy to customize.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# ===== DIRECTORY PATHS =====
# Define all project directories relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # rag_pipeline parent directory
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"  # External dataset folder (if it exists)
DATA_DIR = PROJECT_ROOT / "data"  # Main data directory for corpus, embeddings, eval results
CHROMA_DIR = DATA_DIR / "chroma"  # Vector database storage directory
EVAL_DIR = DATA_DIR / "eval"  # Evaluation set and results directory

# Create directories if they don't exist
for d in (DATA_DIR, CHROMA_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ===== API KEYS AND AUTHENTICATION =====
# Retrieved from environment variables (set in .env file)
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")  # API key for Together AI (LLM + embeddings provider)

# ===== MODEL SELECTION =====
# These models are accessed via Together AI's API
GENERATION_MODEL = "openai/gpt-oss-120b"  # LLM for generating answers and evaluation items
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"  # Embedding model for semantic search
EMBEDDING_DIM = 1024  # Dimension of embeddings (vector size for cosine similarity search)

# ===== TEXT CHUNKING PARAMETERS =====
# How to split long documents into retrievable chunks
CHUNK_SIZE_CHARS = 1400  # Target chunk size in characters (sliding window)
CHUNK_OVERLAP_CHARS = 150  # Overlap between consecutive chunks to preserve context
TOP_K = 5  # Number of chunks to retrieve for each question

# ===== VECTOR DATABASE SETTINGS =====
COLLECTION_NAME = "ai_safety_corpus"  # Name of the Chroma collection for storing embeddings

# ===== DATA SOURCE PATHS =====
# Paths to organization corpora (pre-built document sets)
ORG_CORPORA = {
    "Anthropic": DATASET_ROOT / "Anthropic" / "8 - Corpus" / "pdf_docs.json",
    "DeepMind":  DATASET_ROOT / "DeepMind"  / "8 - Corpus" / "pdf_docs.json",
    "OpenAI":    DATASET_ROOT / "OpenAI"    / "8 - Corpus" / "pdf_docs.json",
}

# Paths to organization article directories (RTF files with press clippings)
ORG_ARTICLE_DIRS = {
    "Anthropic": DATASET_ROOT / "Anthropic" / "9 - Articles",
    "DeepMind":  DATASET_ROOT / "DeepMind"  / "9 - Articles",
    "OpenAI":    DATASET_ROOT / "OpenAI"    / "9 - Articles",
}

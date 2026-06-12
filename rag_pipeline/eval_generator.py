"""Evaluation set generation from corpus chunks using institutional logic matrix.

This module generates (question, reference_answer) pairs for evaluation by:
1. Sampling chunks from the corpus (PDFs and RTF articles)
2. For each chunk, asking the LLM to:
   - Identify ONE elemental category (Y-axis) reflected in the chunk
   - Identify ONE institutional order (X-axis) that characterizes the logic
   - Write a self-contained question whose answer is this (order, category) pair
   - Provide justification grounded in the chunk
3. Validating that the LLM's categorization matches the matrix
4. Writing valid items to JSONL file

This mirrors the methodology in Sarkar & Faik (2026): pattern-matching of
organizational decisions to ideal-type institutional logics.
"""
import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from .config import EVAL_DIR, ORG_CORPORA
from .ingest import chunk_text
from .institutional_logics import CATEGORIES, ORDERS, matrix_as_markdown
from .llm import chat

# Minimum chunk length to consider for evaluation generation
# (smaller chunks don't have enough context to generate meaningful questions)
MIN_CHUNK_CHARS = 600

# System prompt for the LLM: defines its role in generating evaluation items
GEN_SYSTEM = (
    "You generate evaluation items for an AI alignment governance benchmark. "
    "Each item asks which institutional logic (from Thornton & Ocasio's ideal "
    "types) best characterises an organizational decision described in a corpus "
    "excerpt, and along which elemental category."
)

GEN_USER = """Reference matrix (Thornton & Ocasio ideal types):
{matrix}

You will read one excerpt from {org}'s public AI governance corpus and produce
ONE evaluation item.

Excerpt:
\"\"\"{excerpt}\"\"\"

Steps:
1. Identify ONE elemental category from {categories} that is most clearly
   illustrated by the excerpt.
2. Identify ONE institutional order from {orders} that best characterises the
   logic guiding the decision in that category.
3. Write a self-contained question that an analyst could answer WITHOUT seeing
   this excerpt, but that has enough specifics (named program, policy, or
   practice) that the institutional logic can be inferred from {org}'s public
   record. Do not quote the excerpt verbatim; paraphrase.
4. Write a one-sentence reference justification grounded in the excerpt.

If the excerpt is too generic to support a confident judgment, output:
{{"skip": true, "reason": "..."}}

Otherwise output strictly this JSON object (no prose, no markdown fence):
{{
  "question": "...",
  "reference_category": "<one of {categories}>",
  "reference_order": "<one of {orders}>",
  "reference_justification": "..."
}}"""


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response, handling markdown formatting.
    
    Args:
        text: Raw LLM output (may include markdown code blocks)
        
    Returns:
        dict | None: Parsed JSON object, or None if extraction fails
    """
    text = text.strip()
    # Remove markdown code block markers if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    # Find JSON object in text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def sample_chunks(per_org: int, seed: int = 42) -> list[dict]:
    """Sample chunks from corpus to use for evaluation item generation.
    
    Args:
        per_org: Number of chunks to sample from each organization
        seed: Random seed for reproducibility
        
    Returns:
        list[dict]: Sampled chunks with org, filename, and chunk text
    """
    rng = random.Random(seed)
    out = []
    for org, path in ORG_CORPORA.items():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        # Build a pool of sufficiently large chunks
        pool = []
        for doc in docs:
            for chunk in chunk_text(doc.get("text", "")):
                if len(chunk) >= MIN_CHUNK_CHARS:
                    pool.append({"org": org, "filename": doc.get("filename", ""), "chunk": chunk})
        # Shuffle and take top N
        rng.shuffle(pool)
        out.extend(pool[:per_org])
    return out


def generate_eval_set(per_org: int = 20, out_path: Path | None = None) -> Path:
    """Generate evaluation items from sampled corpus chunks.
    
    For each sampled chunk:
    1. Ask LLM to identify institutional order and category
    2. Ask LLM to write a question and justification
    3. Validate against the institutional logic matrix
    4. Write valid items to JSONL file
    
    Args:
        per_org: Number of chunks to sample per organization
        out_path: Output file path (default: EVAL_DIR/eval_set.jsonl)
        
    Returns:
        Path: Path to the generated evaluation set file
    """
    out_path = out_path or (EVAL_DIR / "eval_set.jsonl")
    # Sample chunks from corpus
    samples = sample_chunks(per_org)
    items = []
    cats = ", ".join(CATEGORIES)
    orders = ", ".join(ORDERS)
    matrix_md = matrix_as_markdown()

    # Generate items from samples
    for s in tqdm(samples, desc="generate"):
        prompt = GEN_USER.format(
            matrix=matrix_md, org=s["org"], excerpt=s["chunk"],
            categories=cats, orders=orders,
        )
        # Call LLM to generate evaluation item
        raw = chat(
            [{"role": "system", "content": GEN_SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=600,
        )
        parsed = _extract_json(raw)
        # Skip if LLM flagged as too generic or parsing failed
        if not parsed or parsed.get("skip"):
            continue
        # Validate that the identified order and category are in the matrix
        if parsed.get("reference_order") not in ORDERS:
            continue
        if parsed.get("reference_category") not in CATEGORIES:
            continue
        # Add valid item
        items.append({
            "source_org": s["org"],
            "source_filename": s["filename"],
            "question": parsed["question"],
            "reference_category": parsed["reference_category"],
            "reference_order": parsed["reference_order"],
            "reference_justification": parsed.get("reference_justification", ""),
        })

    # Write items to JSONL file (one JSON object per line)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} eval items -> {out_path}")
    return out_path

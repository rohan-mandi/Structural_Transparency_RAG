"""Stage 2: Generate evaluation question-answer pairs.

This script generates a benchmark of (question, reference_answer) pairs by:
1. Sampling text chunks from the ingested corpus
2. For each chunk, asking the LLM to identify:
   - The institutional order (X-axis) characterizing the decision/practice
   - The elemental category (Y-axis) most clearly illustrated
3. Having the LLM write a self-contained question and justification
4. Validating items against the institutional logic matrix
5. Writing valid items to JSONL file

Usage:
    python scripts/02_generate_eval.py             # Generate 20 questions per org
    python scripts/02_generate_eval.py --per-org 50  # Generate 50 questions per org
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.eval_generator import generate_eval_set

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Number of sampled chunks per organization used to generate evaluation items
    ap.add_argument("--per-org", type=int, default=20,
                    help="number of chunks sampled per org (default 20)")
    args = ap.parse_args()
    # Generate the evaluation set and write it to data/eval/eval_set.jsonl
    generate_eval_set(per_org=args.per_org)

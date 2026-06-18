"""Stage 5: Generate the partial-information, per-cell evaluation set.

The generator is restricted to a ~20% sample of the corpus (the "budget").
For each cell of the Thornton & Ocasio matrix it writes the same number of
questions, grounding them in the limited sample while assigning the canonical
cell as the reference answer. This creates an information asymmetry: the
full-corpus RAG must independently reach the conclusion the partially-informed
generator assigned.

Usage:
    python scripts/05_generate_partial_eval.py                  # 1 question/cell, 20%
    python scripts/05_generate_partial_eval.py --per-cell 2     # 2 questions/cell
    python scripts/05_generate_partial_eval.py --fraction 0.2   # change budget size
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.partial_eval_generator import generate_partial_eval_set

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=1,
                    help="questions generated per matrix cell (default 1 -> 63 items)")
    ap.add_argument("--fraction", type=float, default=0.20,
                    help="fraction of the corpus the generator may reference (default 0.20)")
    args = ap.parse_args()
    generate_partial_eval_set(per_cell=args.per_cell, fraction=args.fraction)

"""Stage 4: Generate the theory evaluation set.

One question per cell of the Thornton & Ocasio ideal-types matrix (7 orders x
9 categories = 63 questions). These questions test institutional logics theory
directly and never reference the AI-lab corpus — they serve as a closed-book
control for the corpus-grounded eval set.

Usage:
    python scripts/04_generate_theory_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.theory_eval_generator import generate_theory_eval_set

if __name__ == "__main__":
    generate_theory_eval_set()

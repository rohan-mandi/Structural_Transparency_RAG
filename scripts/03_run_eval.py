"""Stage 3: Run end-to-end evaluation and report results.

This script runs the complete evaluation pipeline:
1. Loads the evaluation set (questions + reference answers)
2. For each question:
    - Uses RAG to retrieve relevant document chunks
    - Calls the LLM to answer based on retrieved context
    - Matches the candidate answer against the reference using LLM grader
3. Aggregates results and generates accuracy report:
    - Overall accuracy
    - Accuracy by institutional category
    - Accuracy by source organization
4. Exports detailed results to CSV

The evaluation is resumable: if interrupted, rerun to continue from where it left off.

Usage:
     python scripts/03_run_eval.py                  # Run corpus-grounded evaluation
     python scripts/03_run_eval.py --scope-org      # Restrict retrieval to source org
     python scripts/03_run_eval.py --fresh          # Start over (ignore existing results)
     python scripts/03_run_eval.py --theory         # Run the theory eval set (closed-book)
     python scripts/03_run_eval.py --partial         # Run the partial-info eval set (full RAG)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.config import EVAL_DIR
from rag_pipeline.eval_harness import run_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Restrict retrieval to the source organization for each evaluation item
    ap.add_argument("--scope-org", action="store_true",
                    help="restrict retrieval to the source org of each question")
    # Ignore any existing results and rerun evaluation from scratch
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results and start over")
    # Shortcut for the theory eval set (closed-book, no retrieval)
    ap.add_argument("--theory", action="store_true",
                    help="evaluate the theory question set (theory_eval_set.jsonl, "
                         "answered closed-book, results in theory_results.jsonl)")
    # Shortcut for the partial-information eval set (answered with full RAG)
    ap.add_argument("--partial", action="store_true",
                    help="evaluate the partial-info question set (partial_eval_set.jsonl, "
                         "answered with full-corpus RAG, results in partial_results.jsonl)")
    args = ap.parse_args()
    if args.theory and args.partial:
        ap.error("choose at most one of --theory / --partial")
    if args.theory:
        eval_path = EVAL_DIR / "theory_eval_set.jsonl"
        results_path = EVAL_DIR / "theory_results.jsonl"
    elif args.partial:
        eval_path = EVAL_DIR / "partial_eval_set.jsonl"
        results_path = EVAL_DIR / "partial_results.jsonl"
    else:
        eval_path = results_path = None
    # Run the evaluation harness with the chosen options
    run_eval(eval_path=eval_path, results_path=results_path,
             scope_to_source_org=args.scope_org, fresh=args.fresh)

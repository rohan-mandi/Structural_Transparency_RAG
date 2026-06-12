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
     python scripts/03_run_eval.py                  # Run evaluation
     python scripts/03_run_eval.py --scope-org      # Restrict retrieval to source org
     python scripts/03_run_eval.py --fresh          # Start over (ignore existing results)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.eval_harness import run_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Restrict retrieval to the source organization for each evaluation item
    ap.add_argument("--scope-org", action="store_true",
                    help="restrict retrieval to the source org of each question")
    # Ignore any existing results and rerun evaluation from scratch
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results.jsonl and start over")
    args = ap.parse_args()
    # Run the evaluation harness with the chosen options
    run_eval(scope_to_source_org=args.scope_org, fresh=args.fresh)

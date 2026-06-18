"""End-to-end evaluation harness: run RAG, match answers, report results.

This module implements the complete evaluation pipeline:
1. Load evaluation set (questions + reference answers)
2. For each question:
   - Run RAG pipeline to get candidate answer
   - Match candidate against reference using LLM grader
   - Record result (match/no-match + reasoning)
3. Aggregate results and generate report

The evaluation is resumable: if results.jsonl exists, questions already in the
file are skipped (useful for long-running evals that may be interrupted).
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .answer_matcher import match_answer, match_theory_answer
from .config import EVAL_DIR
from .rag_qa import answer_closed_book, answer_question
from .retriever import Retriever


def load_eval_set(path: Path) -> list[dict]:
    """Load evaluation set from JSONL file.

    Args:
        path: Path to JSONL file (one JSON object per line)

    Returns:
        list[dict]: List of evaluation items
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_done(path: Path) -> tuple[set[str], list[dict]]:
    """Load already-completed results to enable resumable evaluation.

    Args:
        path: Path to results JSONL file (if it exists)

    Returns:
        tuple:
            - set[str]: Questions that have already been processed
            - list[dict]: All previously recorded results
    """
    if not path.exists():
        return set(), []
    done_qs, rows = set(), []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_qs.add(row.get("question", ""))
            rows.append(row)
    return done_qs, rows


def run_eval(eval_path: Path | None = None, results_path: Path | None = None,
             scope_to_source_org: bool = False, fresh: bool = False) -> Path:
    """Run the complete evaluation pipeline: RAG -> match -> report.

    This is the main evaluation function. It:
    1. Loads the evaluation set (questions + reference answers)
    2. For each question, runs RAG and matches candidate against reference
    3. Records results (including retrievals and match reasoning)
    4. Generates accuracy report broken down by category and organization

    The process is resumable by default: if results already exist, only new
    questions are evaluated. Use fresh=True to start over.

    Args:
        eval_path: Path to evaluation set JSONL (default: EVAL_DIR/eval_set.jsonl)
        results_path: Path to write results JSONL (default: EVAL_DIR/results.jsonl)
        scope_to_source_org: If True, restrict retrieval to each question's source org
        fresh: If True, ignore existing results and start from scratch

    Returns:
        Path: Path to the results JSONL file
    """
    eval_path = eval_path or (EVAL_DIR / "eval_set.jsonl")
    results_path = results_path or (EVAL_DIR / "results.jsonl")
    items = load_eval_set(eval_path)

    done_qs, prior_rows = (set(), []) if fresh else _load_done(results_path)
    if done_qs:
        print(f"resuming: {len(done_qs)} items already in {results_path.name}, "
              f"{len(items) - len(done_qs)} remaining")

    # Theory items are answered closed-book (no retrieval), so only build the
    # retriever if at least one item actually needs the vector DB.
    needs_retriever = any(it.get("eval_type") != "theory" for it in items)
    retriever = Retriever() if needs_retriever else None
    rows = list(prior_rows)
    mode = "w" if fresh else "a"
    pending = [it for it in items if it["question"] not in done_qs]

    with open(results_path, mode, encoding="utf-8") as f:
        if fresh:
            f.truncate(0)
        for item in tqdm(pending, desc="eval"):
            if item.get("eval_type") == "theory":
                # Closed-book: tests parametric theory knowledge; the corpus
                # contains no institutional-logics theory texts.
                rag = answer_closed_book(item["question"])
                verdict = match_theory_answer(
                    question=item["question"], candidate=rag.answer,
                    reference_answer=item["reference_answer"],
                    reference_order=item["reference_order"],
                    reference_category=item["reference_category"],
                )
            else:
                org_filter = item["source_org"] if scope_to_source_org else None
                rag = answer_question(retriever, item["question"], org=org_filter)
                verdict = match_answer(
                    question=item["question"], candidate=rag.answer,
                    reference_order=item["reference_order"],
                    reference_category=item["reference_category"],
                    reference_justification=item.get("reference_justification", ""),
                )
            row = {
                **item,
                "candidate_answer": rag.answer,
                "retrieved_ids": [c.id for c in rag.retrieved],
                "retrieved_orgs": [c.org for c in rag.retrieved],
                "match": verdict["match"],
                "match_reasoning": verdict["reasoning"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)

    _report(rows, results_path.with_suffix(".csv"))
    return results_path


def _safe_to_csv(df: pd.DataFrame, csv_path: Path) -> Path:
    """Save DataFrame to CSV, with fallback to timestamped filename if locked.

    Useful for when the CSV is open in Excel and can't be overwritten.

    Args:
        df: DataFrame to save
        csv_path: Desired CSV path

    Returns:
        Path: Actual file path used (may be different if original was locked)
    """
    try:
        df.to_csv(csv_path, index=False)
        return csv_path
    except PermissionError:
        alt = csv_path.with_name(f"{csv_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        df.to_csv(alt, index=False)
        print(f"  (note: {csv_path.name} was locked — likely open in Excel. Wrote to {alt.name} instead.)")
        return alt


def _report(rows: list[dict], csv_path: Path) -> None:
    """Generate evaluation report: accuracy metrics and CSV export.

    Computes and prints:
    - Overall answer-matching accuracy
    - Accuracy by reference category and organization
    - Overall accuracy by organization

    Writes detailed results to CSV file for later inspection.

    Args:
        rows: List of evaluation result dicts
        csv_path: Path to write CSV report
    """
    df = pd.DataFrame(rows)
    actual_path = _safe_to_csv(df, csv_path)
    n = len(df)
    acc = df["match"].mean() if n else 0.0
    print(f"\n=== Answer-matching accuracy: {acc:.3f}  (n={n}) ===")
    if not n:
        print(f"\nfull results: {actual_path}")
        return

    if "source_org" in df.columns and df["source_org"].notna().any():
        # Corpus-grounded eval: break down by lab.
        print("\nBy reference category, grouped by lab:")
        by_cat = (df.groupby(["source_org", "reference_category"])["match"]
                    .agg(["mean", "count"]).round(3))
        print(by_cat.to_string())

        print("\nOverall by lab:")
        print(df.groupby("source_org")["match"].agg(["mean", "count"]).round(3).to_string())
    else:
        # Theory eval: break down along the two matrix axes.
        print("\nBy institutional order (X-axis):")
        print(df.groupby("reference_order")["match"].agg(["mean", "count"]).round(3).to_string())
        print("\nBy elemental category (Y-axis):")
        print(df.groupby("reference_category")["match"].agg(["mean", "count"]).round(3).to_string())

    print(f"\nfull results: {actual_path}")

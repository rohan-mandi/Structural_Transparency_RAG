"""Partial-information, per-cell evaluation set.

Design goal (information asymmetry):
- The GENERATOR is shown only a fixed ~20% slice of the corpus. From that
  limited budget it writes questions and assigns each one a matrix cell
  (institutional order x elemental category) as the reference answer.
- The EVALUATED RAG system, by contrast, retrieves over the FULL corpus.

Because the generator never sees the whole dataset, the question is grounded in
real material but the reference answer is NOT derived from the same chunks the
RAG will retrieve. This avoids the circularity of generating a question and its
answer from the identical context, and tests whether the full-corpus RAG
independently reaches the generator's conclusion.

Like the theory set, this produces the SAME number of questions per cell
(default 1 -> 63 items), so coverage of the matrix is uniform rather than
following the corpus's topic distribution.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .config import EVAL_DIR, ORG_CORPORA
from .ingest import chunk_text
from .institutional_logics import CATEGORIES, MATRIX, ORDERS, matrix_as_markdown
from .llm import chat

MIN_CHUNK_CHARS = 600
DEFAULT_FRACTION = 0.20
CONTEXT_K = 4  # chunks shown to the generator per cell
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "from", "have", "has",
    "not", "but", "all", "any", "can", "our", "their", "its", "into", "such",
    "one", "two", "more", "most", "than", "out", "via", "per", "use", "used",
    "no", "identified", "affordances", "in", "literature",
}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS and len(w) > 2]


def sample_budget(fraction: float = DEFAULT_FRACTION, seed: int = 7) -> list[dict]:
    """Deterministically take `fraction` of each org's chunks as the generation
    budget. Sampling per-org keeps all three labs represented even at 20%."""
    rng = random.Random(seed)
    budget = []
    for org, path in ORG_CORPORA.items():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        pool = []
        for doc in docs:
            for chunk in chunk_text(doc.get("text", "")):
                if len(chunk) >= MIN_CHUNK_CHARS:
                    pool.append({"org": org, "filename": doc.get("filename", ""), "chunk": chunk})
        rng.shuffle(pool)
        take = max(1, int(round(len(pool) * fraction)))
        budget.extend(pool[:take])
    return budget


def _score(chunk_tokens: set[str], cell_terms: set[str]) -> int:
    return len(chunk_tokens & cell_terms)


def _rank_for_cell(budget_tokens: list[set[str]], budget: list[dict],
                   order: str, category: str, descriptor: str) -> list[int]:
    cell_terms = set(_tokens(f"{order} {category} {descriptor}"))
    scored = [(_score(budget_tokens[i], cell_terms), i) for i in range(len(budget))]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in scored[:CONTEXT_K]]


GEN_SYSTEM = (
    "You write evaluation questions for an AI-governance benchmark grounded in "
    "Thornton & Ocasio's institutional-logics ideal types. You are given a small "
    "set of excerpts from AI labs' public documents and one target cell of the "
    "matrix. You write self-contained questions whose correct answer is that cell."
)

GEN_USER = """Target cell of the institutional-logics matrix:
- Institutional order (X-axis): {order}
- Elemental category (Y-axis): {category}
- Canonical descriptor for this cell: "{descriptor}"

Reference matrix (for your understanding only — do NOT mention it in questions):
{matrix}

Excerpts available to you (a limited sample of the corpus):
{context}

Write exactly {per_cell} self-contained question(s) about a real AI-lab
decision, policy, or practice that would exemplify the {order} institutional
logic along the "{category}" category. Requirements:
- Use the excerpts as background. Prefer grounding each question in something
  concrete they suggest (a named policy, program, governance practice, or
  stated commitment); paraphrase, do not quote verbatim.
- If the excerpts only weakly relate to this cell, STILL write a question. Lean
  on the named lab(s) in the excerpts and ask about the aspect of their conduct
  that, if present in the corpus, would reflect the {order} logic for this
  category. Never refuse and never skip — every cell must get a question.
- Each question must be answerable by an analyst who can read the lab's public
  record, and its intended answer is the {order} logic ({descriptor}).
- Do NOT reveal the answer or name the institutional order inside the question.

Output strictly this JSON (no prose, no markdown fence):
{{"items": [{{"question": "...", "reference_justification": "...", "grounding": "strong|weak"}}]}}"""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _generate_for_cell(prompt: str, per_cell: int, attempts: int = 3) -> list[dict]:
    """Call the model for one cell, retrying on unparseable output. Returns the
    list of question dicts, or [] if every attempt failed to parse."""
    for i in range(attempts):
        raw = chat(
            [{"role": "system", "content": GEN_SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.3 + 0.2 * i, max_tokens=700,
        )
        parsed = _extract_json(raw)
        if parsed and parsed.get("items"):
            return parsed["items"]
    return []


def _fallback_question(order: str, category: str, org: str) -> str:
    who = org or "the AI lab"
    return (f"Based on {who}'s public governance record, what does their approach "
            f"to '{category.lower()}' reveal about the institutional logic guiding "
            f"that aspect of their conduct?")


def generate_partial_eval_set(per_cell: int = 1, fraction: float = DEFAULT_FRACTION,
                              out_path: Path | None = None) -> Path:
    out_path = out_path or (EVAL_DIR / "partial_eval_set.jsonl")
    budget = sample_budget(fraction)
    budget_tokens = [set(_tokens(b["chunk"])) for b in budget]
    matrix_md = matrix_as_markdown()
    print(f"generation budget: {len(budget)} chunks (~{fraction:.0%} of corpus)")

    items = []
    cells = [(cat, order) for cat in CATEGORIES for order in ORDERS]
    for category, order in tqdm(cells, desc="partial-gen"):
        descriptor = MATRIX[category][order]
        idxs = _rank_for_cell(budget_tokens, budget, order, category, descriptor)
        ctx_chunks = [budget[i] for i in idxs]
        context = "\n\n".join(
            f"[{j+1}] (org={c['org']}, file={c['filename']})\n{c['chunk']}"
            for j, c in enumerate(ctx_chunks)
        )
        prompt = GEN_USER.format(
            order=order, category=category, descriptor=descriptor,
            matrix=matrix_md, context=context, per_cell=per_cell,
        )
        # Every cell must yield questions. Retry parse failures a couple times;
        # if the model still won't return valid JSON, fall back to a templated
        # question so the cell is never dropped.
        ctx_orgs = [c["org"] for c in ctx_chunks]
        primary_org = Counter(ctx_orgs).most_common(1)[0][0] if ctx_orgs else ""
        questions = _generate_for_cell(prompt, per_cell)
        if not questions:
            questions = [{
                "question": _fallback_question(order, category, primary_org),
                "reference_justification": "Fallback question (model returned no parseable item).",
                "grounding": "weak",
            }]
        for q in questions[:per_cell]:
            qtext = (q.get("question") or "").strip() or _fallback_question(order, category, primary_org)
            items.append({
                "eval_type": "partial",
                "source_org": primary_org,
                "question": qtext,
                "reference_category": category,
                "reference_order": order,
                "reference_justification": q.get("reference_justification", ""),
                "grounding": q.get("grounding", ""),
                "context_files": [c["filename"] for c in ctx_chunks],
                "context_orgs": ctx_orgs,
                "generation_fraction": fraction,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} partial eval items -> {out_path}")
    return out_path

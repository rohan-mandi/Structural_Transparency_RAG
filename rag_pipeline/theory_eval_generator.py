"""Theory evaluation set: one question per cell of the Thornton & Ocasio matrix.

Unlike eval_generator.py, these questions do NOT reference the AI-lab corpus at
all. They test knowledge of institutional logics theory itself, anchored to the
ideal-types matrix (7 institutional orders x 9 elemental categories) and the
academic literature it comes from. This acts as a closed-book control: it
separates "does the model know the theory" from "can RAG ground the theory in
lab documents".

Reference answers are the canonical cell descriptors, so the answer matcher
has an unambiguous ground truth (the condition Chandak et al. 2025 show makes
LLM matching reliable).
"""
import json
from pathlib import Path

from tqdm import tqdm

from .config import EVAL_DIR
from .institutional_logics import CATEGORIES, MATRIX, ORDERS, matrix_as_markdown
from .llm import chat

# Academic sources on institutional logics used to ground question phrasing.
# These are attached to every generated item for citation/traceability.
ACADEMIC_SOURCES = [
    {
        "citation": "Thornton, P.H. & Ocasio, W. (2008). Institutional Logics. "
                    "In Greenwood et al. (Eds.), The SAGE Handbook of Organizational "
                    "Institutionalism (Ch. 3).",
        "url": "http://patriciathornton.com/files/9781412931236-Ch03.pdf",
    },
    {
        "citation": "Thornton, P.H., Ocasio, W., & Lounsbury, M. (2012). The "
                    "Institutional Logics Perspective: A New Approach to Culture, "
                    "Structure and Process. Oxford University Press. Ch. 3: Defining "
                    "the Interinstitutional System.",
        "url": "https://academic.oup.com/book/35363/chapter/300986180",
    },
    {
        "citation": "Friedland, R. & Alford, R.R. (1991). Bringing Society Back In: "
                    "Symbols, Practices, and Institutional Contradictions. In Powell & "
                    "DiMaggio (Eds.), The New Institutionalism in Organizational "
                    "Analysis, pp. 232-267. University of Chicago Press.",
        "url": "https://www.researchgate.net/publication/238198697_Bringing_Society_Back_In_Symbols_Practices_and_Institutional_Contradictions",
    },
    {
        "citation": "Reay, T. & Jones, C. (2016). Qualitatively capturing "
                    "institutional logics. Strategic Organization, 14(4), 441-454.",
        "url": "https://journals.sagepub.com/doi/10.1177/1476127015589981",
    },
]

GEN_SYSTEM = (
    "You write graduate-level exam questions about institutional logics theory "
    "(Thornton & Ocasio's ideal types of institutional orders). Questions must "
    "be answerable from the theory literature alone — never reference any AI "
    "company, dataset, or document corpus."
)

GEN_USER = """Full ideal-types matrix for context:
{matrix}

Target cell:
- Institutional order (X-axis): {order}
- Elemental category (Y-axis): {category}
- Canonical cell content: "{descriptor}"

Grounding literature: Thornton & Ocasio (2008); Thornton, Ocasio & Lounsbury
(2012); Friedland & Alford (1991).

Write ONE exam-style question whose correct answer is the canonical cell
content above. Requirements:
- Self-contained: an expert who knows the institutional logics literature can
  answer it without seeing the matrix.
- Ask what characterises the {category} of the {order} institutional order
  (phrase it naturally; vary phrasing, e.g. "According to Thornton and
  Ocasio's ideal types..." or "In the interinstitutional system...").
- Do NOT embed the answer in the question.
- Do NOT mention any company, AI system, or dataset.

Output strictly this JSON object (no prose, no markdown fence):
{{"question": "..."}}"""


def _extract_json(text: str) -> dict | None:
    import re
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate_theory_eval_set(out_path: Path | None = None) -> Path:
    """Generate one question per matrix cell (63 total, including the two
    'no identified affordances' cells, whose reference answer is that none
    have been identified in the literature)."""
    out_path = out_path or (EVAL_DIR / "theory_eval_set.jsonl")
    matrix_md = matrix_as_markdown()
    items = []

    cells = [(cat, order) for cat in CATEGORIES for order in ORDERS]
    for category, order in tqdm(cells, desc="theory-gen"):
        descriptor = MATRIX[category][order]
        prompt = GEN_USER.format(
            matrix=matrix_md, order=order, category=category, descriptor=descriptor,
        )
        raw = chat(
            [{"role": "system", "content": GEN_SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300,
        )
        parsed = _extract_json(raw)
        if not parsed or not parsed.get("question"):
            print(f"  [skip] generation failed for ({order}, {category})")
            continue
        items.append({
            "eval_type": "theory",
            "question": parsed["question"].strip(),
            "reference_category": category,
            "reference_order": order,
            "reference_answer": descriptor,
            "sources": [s["citation"] for s in ACADEMIC_SOURCES],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} theory eval items -> {out_path}")
    return out_path

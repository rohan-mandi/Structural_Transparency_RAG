"""Answer Matcher: LLM-based grading of free-form responses (Chandak et al. 2025).

Given a question, a candidate free-form response, and a reference answer, uses an
LLM to determine if the candidate matches the reference. This approach ("LLM-as-
reference-aware-judge") is shown by Chandak et al. to align better with human
grading than LLM-as-judge without a reference.

The matcher evaluates whether the candidate response correctly identifies:
1. The institutional order (X-axis of the matrix)
2. The elemental category (Y-axis of the matrix)
3. Consistent reasoning with the reference justification
"""
import json
import re

from .institutional_logics import MATRIX
from .llm import chat

# System prompt: instructs the LLM how to grade
MATCH_SYSTEM = (
    "You grade free-form responses against a reference answer. You are strict, "
    "consistent, and ignore stylistic differences. The reference answer is "
    "authoritative."
)

MATCH_USER = """Question:
{question}

Reference answer:
- Institutional order (X-axis): {reference_order}
- Elemental category (Y-axis): {reference_category}
- Canonical descriptor for this cell: "{cell_descriptor}"
- Justification: {reference_justification}

Candidate response:
\"\"\"{candidate}\"\"\"

Grading rules:
- The candidate matches if it names the SAME institutional order ({reference_order})
  as the dominant logic, AND its reasoning is consistent with the elemental
  category ({reference_category}) or its canonical descriptor.
- Paraphrases are fine. Exact wording is not required.
- If the candidate names a different order, or hedges across several orders
  without committing to {reference_order}, it does NOT match.
- A candidate that says the excerpt is insufficient or refuses to answer does
  NOT match.

Output strictly this JSON object (no prose, no markdown):
{{"match": true|false, "reasoning": "one short sentence"}}"""


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response, removing markdown formatting if present.
    
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


def match_answer(
    *, question: str, candidate: str,
    reference_order: str, reference_category: str,
    reference_justification: str = "",
) -> dict:
    """Grade a candidate answer against a reference answer using the LLM.
    
    Args:
        question: The original question
        candidate: The free-form answer to grade
        reference_order: The correct institutional order (ground truth)
        reference_category: The correct elemental category (ground truth)
        reference_justification: Optional brief explanation of the reference answer
        
    Returns:
        dict with keys:
            'match': bool - whether the answer matches the reference
            'reasoning': str - brief explanation of the grading decision
            'raw': str - raw LLM output (for debugging)
    """
    # Get the canonical descriptor from the institutional logic matrix
    cell = MATRIX.get(reference_category, {}).get(reference_order, "")
    # Format the user prompt with specific reference answer details
    user = MATCH_USER.format(
        question=question, candidate=candidate,
        reference_order=reference_order, reference_category=reference_category,
        cell_descriptor=cell, reference_justification=reference_justification,
    )
    # Call LLM to grade the answer
    raw = chat(
        [{"role": "system", "content": MATCH_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.0, max_tokens=200,
    )
    # Parse JSON response from LLM
    parsed = _extract_json(raw) or {}
    return {
        "match": bool(parsed.get("match", False)),
        "reasoning": parsed.get("reasoning", "").strip(),
        "raw": raw,
    }

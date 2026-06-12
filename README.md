# RAG + Answer-Matching Evaluation

A RAG pipeline over the Anthropic / DeepMind / OpenAI AI-safety corpora, with
an evaluation harness modeled on Chandak et al. (2025, *Answer Matching
Outperforms Multiple Choice*) and anchored in the Institutional Logics matrix
of Sarkar & Faik (2026, *Structural Transparency*).

## What it does

1. **Ingest** the pre-built corpora (`Anthropic/8 - Corpus/`, `DeepMind/8 - Corpus/`,
   `OpenAI/8 - Corpus/`), chunk them, embed with Together's BGE-Large, and
   persist into a local **Chroma** vector store.
2. **Answer** questions via a RAG chain: retrieve top-k chunks, prompt
   `openai/gpt-oss-120b` on Together with the excerpts, and return a free-form
   response (the *candidate* answer).
3. **Generate** an eval set by sampling chunks per org and asking gpt-oss-120b
   to produce questions whose reference answers are an (institutional order,
   elemental category) pair from Thornton & Ocasio's ideal-types matrix
   (Structural Transparency Fig. 2).
4. **Match** each candidate answer to the reference using gpt-oss-120b as the
   *matcher* — given the reference, not as an LLM-as-judge.
5. **Report** accuracy overall, by source org, by institutional order, and by
   elemental category.

## Setup

```bash
cd "RAG_workflow_tutorial"
pip install -r requirements.txt
copy .env.example .env
# edit .env and paste your TOGETHER_API_KEY
```

## Run

```bash
python scripts/01_ingest.py --reset
python scripts/02_generate_eval.py --per-org 20
python scripts/03_run_eval.py
# add --scope-org to restrict retrieval to the source org per question
```

Outputs land in `data/`:

- `data/chroma/` — persistent vector DB
- `data/eval/eval_set.jsonl` — generated Q&A pairs
- `data/eval/results.jsonl` and `results.csv` — per-item candidate, retrieved
  chunks, match verdict, and matcher reasoning

## Layout

```
rag_pipeline/
  config.py                 paths, model names, chunking params
  institutional_logics.py   Thornton & Ocasio matrix (7 orders x 9 categories)
  ingest.py                 chunk + embed + persist to Chroma
  retriever.py              Chroma retrieval wrapper
  llm.py                    Together chat wrapper
  rag_qa.py                 retrieve -> prompt -> candidate answer
  eval_generator.py         sample chunks -> generate reference Q&A
  answer_matcher.py         candidate vs reference -> match verdict
  eval_harness.py           end-to-end runner + reporting
scripts/
  01_ingest.py 02_generate_eval.py 03_run_eval.py
```

## Notes

- **Security**: never commit `.env`. Rotate the Together key if it has been
  shared in chat.
- **Costs**: with default settings (per-org=20), ingestion does ~thousands of
  embedding calls and evaluation does ~60 generation + ~60 matcher calls.
- **Why anchor the eval to the IL matrix?** Sarkar & Faik's C1 component
  specifies pattern-matching organizational decisions to ideal-type logics.
  Using the matrix as the reference space gives the answer matcher a
  well-defined ground truth, which is exactly the condition Chandak et al.
  show pushes matcher agreement up to inter-annotator levels.

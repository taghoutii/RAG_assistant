# ml-papers-rag

An evaluated retrieval-augmented question-answering system over a curated corpus of AI/ML research papers.
Answers cite the exact passages they rely on, and the system abstains when the corpus lacks evidence.

> Work in progress. The full README (results table, architecture, findings) is written in Phase 8.

Built in plain Python, deliberately without LangChain or LlamaIndex, so every step of the pipeline is visible.

## Setup (Windows PowerShell)

Requires Python 3.11 (`py -0p` lists installed versions).

```powershell
cd C:\Users\tagho\ml-papers-rag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

If `Activate.ps1` is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Chunking and indexing

- **Section-aware chunks, sized in tokens.** Chunks never cross a section boundary. Their size is measured with the embedding model's own tokenizer (`BAAI/bge-small-en-v1.5`), so every embedded sequence fits the model's 512-token window: the build checks that the short header, the chunk text and the special tokens never exceed it.
- **Different headers for the two indexes, on purpose.** Each chunk is indexed with a header naming its paper and section, but the two retrievers see different headers:
  - **BM25** gets the full header: the full paper title and the full section path. BM25 has no length limit, and the extra words help keyword matching.
  - **Dense** retrieval gets a short header: the title before its colon plus the last section heading, capped at 32 tokens. Every header token is taken from the 512-token budget the chunk itself needs.

  The chunk text without any header is what is quoted, cited and checked against the evaluation set.

```powershell
python -m ragpapers.chunking.build            # chunk files + token-budget and quote checks
python -m ragpapers.evaluation.run_retrieval  # indexes + retrieval evaluation on the dev split
```

## Models (free tiers only)

| Role | Model | Free-tier limits (per model) |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) | none |
| Generator | `openai/gpt-oss-120b` on Groq, top 8 plain-hybrid chunks as context | 30 RPM, 1K RPD, 8K TPM, 200K TPD |
| Judge | Gemini 3.5 Flash-Lite (Google AI Studio) | 15 RPM, 250K TPM, 500 RPD |
| Fallback judge | `qwen/qwen3.8-27b` on Groq (separate per-model budget) | 30 RPM, 1K RPD, 8K TPM, 200K TPD |

- **Why gpt-oss:** Llama models left Groq's free tier on 2026-08-16; gpt-oss-120b is the largest text model it still serves.
- **Why a judge from another provider:** it has its own quota, and a different model family limits a judge's bias toward its own generator's answers. Gemini's Flash models are capped at 20 requests a day on the free tier, too few for one evaluation run, so the judge is Flash-Lite.
- **Fallback judge, decided before any grading:** if the primary judge agrees poorly with 20 hand-graded answers, the fallback is graded on the same 20 and the judge that agrees better with the hand grades is kept.
- **Why 8 chunks:** retrieval recall on the dev set is 0.51 / 0.61 / 0.72 at 3 / 5 / 8 chunks; a full dev run at 8 chunks uses about 137K of the generator's 200K daily tokens.

Limits as listed on the free-tier consoles in October 2026; they change, so check yours. All values are in `configs/default.yaml`.

## Checks

```powershell
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

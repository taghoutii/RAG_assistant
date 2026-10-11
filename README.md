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

## Checks

```powershell
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

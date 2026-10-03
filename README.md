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

## Checks

```powershell
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

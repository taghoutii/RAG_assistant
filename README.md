# ml-papers-rag

An evaluated retrieval-augmented question-answering system over a curated corpus of AI/ML research papers.
Answers cite the exact passages they rely on, and the system abstains when the corpus lacks evidence.

> Work in progress. The full README (architecture, generation results, findings) is written in Phase 8.

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
python -m ragpapers.chunking.build                 # chunk file + token-budget and quote checks
python -m ragpapers.evaluation.evaluate_retrieval  # indexes + the retrieval table on the dev split
```

## Retrieval results (dev)

30 answerable dev questions; each setup returns its top 10 chunks. Relevance is judged against the gold evidence quotes and their approved alternatives (see [eval/README.md](eval/README.md)). Brackets: 95% bootstrap interval over questions. Full table with paired differences: [reports/retrieval.md](reports/retrieval.md).

| Setup | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 | Latency p50 |
|---|---|---|---|---|---|---|
| BM25 | 0.494 [0.33, 0.65] | 0.517 [0.36, 0.67] | 0.656 [0.51, 0.79] | 0.499 [0.35, 0.64] | 0.443 [0.31, 0.57] | 8 ms |
| Dense | 0.444 [0.28, 0.62] | 0.517 [0.36, 0.68] | 0.644 [0.51, 0.80] | 0.400 [0.27, 0.54] | 0.362 [0.26, 0.47] | 23 ms |
| **Hybrid (RRF)** | 0.511 [0.34, 0.68] | **0.611** [0.46, 0.77] | **0.772** [0.63, 0.90] | **0.533** [0.39, 0.68] | **0.486** [0.36, 0.61] | 32 ms |
| Hybrid + reranker | 0.517 [0.36, 0.68] | 0.594 [0.43, 0.75] | 0.678 [0.53, 0.83] | 0.501 [0.36, 0.65] | 0.449 [0.33, 0.57] | 1375 ms |

Latencies measured on an Intel i7-11370H laptop, CPU only.

## Key design decisions

- **448-token chunks with 64-token overlap:** chunks large enough to carry complete passages, kept under the 478-token body budget (512-token window minus 2 special tokens and the 32-token header), so no chunk is truncated when embedded.
- **Hybrid retrieval with Reciprocal Rank Fusion, k = 60:** BM25 and dense rankings are fused by rank only, so their incompatible score scales never meet; k = 60 is the standard value (Cormack et al., 2009), with nothing to tune.
- **No reranker in the final system:** the MiniLM cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, over the top 20 fused chunks) does not beat plain hybrid and adds over a second per query on CPU; it stays in the table as a measured alternative.

## Models (free tiers only)

| Role | Model | Free-tier limits (per model) |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) | none |
| Generator | `openai/gpt-oss-120b` on Groq, top 8 plain-hybrid chunks as context | 30 RPM, 1K RPD, 8K TPM, 200K TPD |
| Judge | `qwen/qwen3.8-27b` on Groq (separate per-model budget) | 30 RPM, 1K RPD, 8K TPM, 200K TPD |

- **Why gpt-oss:** Llama models left Groq's free tier on 2026-08-16; gpt-oss-120b is the largest text model it still serves.
- **How the judge was chosen:** the judge was chosen by a rule fixed before grading; Gemini 3.5 Flash-Lite was tested first and failed the agreement threshold, so the predefined fallback, qwen, was kept.
- **Why 8 chunks:** retrieval recall on the dev set is 0.51 / 0.61 / 0.72 at 3 / 5 / 8 chunks; a full dev run at 8 chunks uses about 137K of the generator's 200K daily tokens.

Limits as listed on the free-tier consoles in October 2026; they change, so check yours. All values are in `configs/default.yaml`.

## Checks

```powershell
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

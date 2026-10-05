# Evaluation set

`questions.jsonl` is the answer key for every experiment in this project. It is finalized
**before** retrieval is tuned, so the system is measured against questions that were not
chosen to suit it.

| File | What it is |
|---|---|
| `questions.jsonl` | The evaluation set: one question per line, each one reviewed before inclusion. |
| `splits.json` | The frozen dev/test split (generated once, never edited). |
| `candidates.jsonl` | Draft questions proposed by the assistant. Each needs approval before it is moved into `questions.jsonl`. |

## Format

One JSON object per line:

```json
{"id": "q001", "question": "...?", "type": "factual",
 "reference_answer": "...",
 "evidence": [{"paper_id": "2106.09685", "quote": "exact supporting span"}],
 "notes": "optional"}
```

| Field | Rule |
|---|---|
| `id` | `q` + 3 digits, unique. |
| `question` | Ends with `?`. Phrase it as a user would, not by copying the paper's wording. |
| `type` | `factual`, `comparison` or `unanswerable`. |
| `reference_answer` | A short correct answer. For `unanswerable`, exactly the abstention sentence from `configs/default.yaml` (`abstention_text`). |
| `evidence` | `factual`: at least 1 quote, all from **one** paper. `comparison`: quotes from **at least 2** papers. `unanswerable`: `[]`. |
| `notes` | Evidence only, no authorship or review comments. Answerable questions: the source section(s), plus anything a checker needs (e.g. a number the paper states in two ways). Unanswerable questions: the kind (near-miss, false premise, off-corpus), the `find_quote` search(es) over the whole corpus, and the best score found. |

## Evidence quotes

A quote is how the evaluation recognises a relevant chunk, whatever the chunking. A retrieved chunk counts as relevant if it contains an evidence quote, and each quote is credited only once, at its highest-ranked match.

- **Copy it exactly** from the parsed text: use `find_quote` (below) or the previews in `data/processed/<id>.md`. Case, curly quotes, dashes, `$` math signs and citation markers (`[ref]`, `[12]`) are ignored when matching. Everything else must match.
- **5–50 words.** Shorter quotes match too many places; longer ones risk being split across two chunks.
- **One block.** Do not join text from two paragraphs, or from a paragraph and a table.
- **It must contain the fact that answers the question**, not just the topic.
- Prefer the main text over the abstract when both state the fact. Answers found only in the abstract make retrieval look easier than it is.

## Question mix

Target: **60–80 questions**, about **55% factual, 25% comparison, 20% unanswerable**.

Write unanswerable questions of three kinds, and say which in `notes`:
- **Near-miss** (the most valuable): a plausible detail the paper on that topic does not state. Example: Mistral 7B's learning rate.
- **False premise:** the question assumes something the paper does not say.
- **Off-corpus:** a topic none of the 44 papers covers.

Before calling a question unanswerable, search for its key terms with `find_quote` across the whole corpus.

## Workflow (PowerShell, from the repo root)

```powershell
# 1. Find exact text to quote (whole corpus, or one paper)
python -m ragpapers.evaluation.find_quote "trainable parameters by 10,000 times"
python -m ragpapers.evaluation.find_quote "learning rate" --paper 2310.06825 --top 10

# 2. Validate as you write: schema, quotes found in the corpus, duplicates, size, type mix
python -m ragpapers.evaluation.questions
python -m ragpapers.evaluation.questions eval/candidates.jsonl

# 3. Once the set is final and has 0 errors: freeze the stratified dev/test split (once!)
python -m ragpapers.evaluation.split
```

The validator reports **errors**, which must be fixed (the split refuses to run while any remain), and **warnings**, which you should look at (fuzzy-only quotes, quote length, size, mix). In CI, only the schema is checked, because the parsed corpus is not committed.

## Dev and test

- **Dev (50%):** used for every tuning decision in Phases 3–6.
- **Test (50%):** used once, for the final report in Phase 6.

With about 70 questions, each half has about 35, including about 7 unanswerable ones. A 50/50 split keeps the held-out test set large enough for the final numbers to mean something. The cost is a smaller dev set to tune on.

Each question type is split separately, with the seed from the config (`eval.test_fraction`, `seed`), so both sets have the same mix. After `splits.json` exists, do not move questions between sets, and do not edit test questions after looking at results.

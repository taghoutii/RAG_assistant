"""Phase 3 retrieval experiment (token-sized chunks): chunk size x retriever, on the dev set.

Grid: every chunk configuration in `chunking.compare` x {bm25, dense}, appendices included and no
query instruction (the word-based Phase 3 run found no measurable effect from either). The
word-based Phase 3 results (reports/results/phase3/) are added as reference rows, so word and
token sizing can be compared directly.

For every setup, each answerable dev question is run through the retriever, the top-n chunks are
scored with `credit_quotes` + Recall@k / reciprocal rank / nDCG@k (evaluation/metrics.py), and
the means get 95% bootstrap confidence intervals (resampling questions). Each setup writes one
JSON file to reports/results/phase3_tokens/, and reports/phase3_tokens_retrieval.md (results +
paired comparisons) is generated from those files.

Usage (PowerShell, from the repo root):
    python -m ragpapers.chunking.build            # chunk files first (checks the token budget)
    python -m ragpapers.evaluation.run_retrieval  # first run embeds (~25 min on CPU), then cached
"""

import json
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from ragpapers.chunking.build import chunks_path, config_name, load_chunks
from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.metrics import credit_quotes, ndcg_at_k, recall_at_k, reciprocal_rank
from ragpapers.evaluation.questions import evidence_groups, load_questions
from ragpapers.evaluation.split import load_split
from ragpapers.index.bm25 import BM25Index
from ragpapers.index.dense import DenseIndex, Embedder, load_or_compute_embeddings

PHASE = "phase3_tokens"


# --- scoring -----------------------------------------------------------------------------------


def score_question(
    retrieved: list[dict], quote_groups: list[list[str]], ks: list[int], threshold: float
) -> dict:
    """All metrics for one question, given its retrieved chunks (best first) and evidence groups."""
    credited = credit_quotes([c["text"] for c in retrieved], quote_groups, threshold)
    scores = {"rr": reciprocal_rank(credited)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(credited, len(quote_groups), k)
        scores[f"ndcg@{k}"] = ndcg_at_k(credited, len(quote_groups), k)
    return {"credited": credited, "scores": scores}


def bootstrap_ci(values: list[float], n_samples: int, seed: int) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean, resampling the values with replacement."""
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_samples))
    return means[int(0.025 * n_samples)], means[int(0.975 * n_samples) - 1]


def aggregate(per_question: list[dict], n_samples: int, seed: int) -> dict:
    """{metric: {"mean", "ci_low", "ci_high"}} over questions; "rr" is reported as "mrr"."""
    out = {}
    for metric in per_question[0]["scores"]:
        values = [q["scores"][metric] for q in per_question]
        low, high = bootstrap_ci(values, n_samples, seed)
        name = "mrr" if metric == "rr" else metric
        out[name] = {"mean": statistics.fmean(values), "ci_low": low, "ci_high": high}
    return out


def paired_difference(
    a: dict, b: dict, metric: str, n_samples: int, seed: int
) -> tuple[float, float, float]:
    """Mean of (a - b) per question, with a 95% paired bootstrap interval.

    Both results must cover the same questions. Resampling the per-question differences is far
    more sensitive than comparing two separate confidence intervals.
    """
    key = "rr" if metric == "mrr" else metric
    a_scores = {q["id"]: q["scores"][key] for q in a["per_question"]}
    b_scores = {q["id"]: q["scores"][key] for q in b["per_question"]}
    if a_scores.keys() != b_scores.keys():
        raise ValueError(f"{a['name']} and {b['name']} were run on different questions")
    diffs = [a_scores[i] - b_scores[i] for i in sorted(a_scores)]
    low, high = bootstrap_ci(diffs, n_samples, seed)
    return statistics.fmean(diffs), low, high


# --- running one setup -------------------------------------------------------------------------


def run_setup(
    name: str, setup: dict, retrieve, questions: list[dict], cfg: dict, seed: int
) -> dict:
    """Run `retrieve(question_text, k)` for every question and collect metrics."""
    per_question, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        retrieved = retrieve(q["question"], cfg["top_n"])
        latencies.append(1000 * (time.perf_counter() - start))
        result = score_question(
            retrieved, evidence_groups(q), cfg["ks"], cfg["quote_match_threshold"]
        )
        per_question.append(
            {
                "id": q["id"],
                "type": q["type"],
                "retrieved": [c["chunk_id"] for c in retrieved],
                **result,
            }
        )
    return {
        "name": name,
        "phase": PHASE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": cfg["split"],
        "n_questions": len(questions),
        "setup": setup,
        "metrics": aggregate(per_question, cfg["bootstrap_samples"], seed),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "max": max(latencies),
        },
        "per_question": per_question,
    }


def write_result(result: dict, results_dir: Path) -> Path:
    path = results_dir / result["phase"] / f"{result['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


# --- tables ------------------------------------------------------------------------------------


def fmt(metric: dict, with_ci: bool = False) -> str:
    text = f"{metric['mean']:.3f}"
    if with_ci:
        text += f" [{metric['ci_low']:.2f}, {metric['ci_high']:.2f}]"
    return text


def unit_label(result: dict) -> str:
    unit = result["setup"].get("unit", "words")  # Phase 3 word-based results have no "unit"
    return f"{unit} (reference)" if result.get("reference") else unit


def make_table(results: list[dict]) -> str:
    """Markdown table, one row per setup: reference rows first, then retriever and chunk size."""
    order = {"bm25": 0, "dense": 1, "dense+instruction": 2}
    rows = sorted(
        results,
        key=lambda r: (
            not r.get("reference", False),
            order.get(r["setup"]["retriever"], 9),
            r["setup"]["chunk_size"],
            not r["setup"]["include_appendix"],
        ),
    )
    lines = [
        "| Retriever | Unit | Chunk / overlap | Appendix | Recall@3 | Recall@5 "
        "| Recall@10 [95% CI] | MRR [95% CI] | nDCG@10 | Latency p50 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s, m = r["setup"], r["metrics"]
        lines.append(
            f"| {s['retriever']} | {unit_label(r)} | {s['chunk_size']} / {s['overlap']} "
            f"| {'yes' if s['include_appendix'] else 'no'} "
            f"| {fmt(m['recall@3'])} | {fmt(m['recall@5'])} | {fmt(m['recall@10'], True)} "
            f"| {fmt(m['mrr'], True)} | {fmt(m['ndcg@10'])} | {r['latency_ms']['p50']:.0f} |"
        )
    n = rows[0]["n_questions"] if rows else 0
    header = (
        f"Retrieval on the **{rows[0]['split'] if rows else 'dev'}** split: {n} answerable "
        "questions (unanswerable questions have no evidence and are excluded). Brackets: 95% "
        "bootstrap confidence interval over questions.\n\n"
    )
    return header + "\n".join(lines) + "\n"


def comparisons_table(pairs: list[tuple[str, dict, dict]], n_samples: int, seed: int) -> str:
    """Markdown table of paired differences (A - B) for Recall@10 and MRR."""
    lines = [
        "| Comparison (A - B) | Recall@10 diff [95% CI] | MRR diff [95% CI] |",
        "|---|---|---|",
    ]
    for label, a, b in pairs:
        cells = []
        for metric in ("recall@10", "mrr"):
            diff, low, high = paired_difference(a, b, metric, n_samples, seed)
            mark = " *" if low > 0 or high < 0 else ""
            cells.append(f"{diff:+.3f} [{low:+.2f}, {high:+.2f}]{mark}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    note = "\nPaired bootstrap over the same questions; * = the interval excludes 0.\n"
    return "\n".join(lines) + "\n" + note


# --- main --------------------------------------------------------------------------------------


def main() -> None:
    config = load_config()
    cfg = config["retrieval_eval"]
    seed = config["seed"]
    emb_cfg = config["embedding"]
    chunking = config["chunking"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]

    split_ids = load_split(REPO_ROOT / config["paths"]["eval_splits"], cfg["split"])
    questions = [
        q
        for q in load_questions(REPO_ROOT / config["paths"]["eval_questions"])
        if q["id"] in split_ids and q["evidence"]
    ]
    print(f"{len(questions)} answerable {cfg['split']} questions")

    embedder = Embedder(emb_cfg["model"], emb_cfg["batch_size"])
    client = QdrantClient(path=str(REPO_ROOT / config["index"]["qdrant_path"]))
    results: dict[str, dict] = {}

    for setting in chunking["compare"]:
        size, overlap = setting["chunk_size"], setting["overlap"]
        cname = config_name(chunking["unit"], size, overlap)
        chunks = load_chunks(chunks_path(config, chunking["unit"], size, overlap))
        print(f"\n{cname}: {len(chunks)} chunks")

        cache = REPO_ROOT / config["index"]["embeddings_dir"] / f"{cname}.npy"
        vectors = load_or_compute_embeddings(
            chunks, embedder.encode_passages, emb_cfg["model"], cache
        )
        dense = DenseIndex.build(client, f"chunks_{cname}", chunks, vectors)
        bm25 = BM25Index(chunks, **config["bm25"])

        def bm25_retrieve(query, k, index=bm25):
            return [c for c, _ in index.search(query, k)]

        def dense_retrieve(query, k, index=dense):
            return [c for c, _ in index.search(embedder.encode_query(query), k)]

        for retriever, fn in (("bm25", bm25_retrieve), ("dense", dense_retrieve)):
            setup = {
                "retriever": retriever,
                "unit": chunking["unit"],
                "tokenizer": chunking["tokenizer"],
                "chunk_size": size,
                "overlap": overlap,
                "include_appendix": True,
                "n_chunks_searched": len(chunks),
                "indexed_text": (
                    "full header (title > full section path) + chunk text"
                    if retriever == "bm25"
                    else f"capped dense header ({chunking['dense_header']}) + chunk text"
                ),
                "embedding_model": emb_cfg["model"] if retriever == "dense" else None,
                "query_instruction": None,
                "bm25": config["bm25"] if retriever == "bm25" else None,
                "top_n": cfg["top_n"],
                "quote_match_threshold": cfg["quote_match_threshold"],
            }
            name = f"{retriever}_{cname}"
            result = run_setup(name, setup, fn, questions, cfg, seed)
            write_result(result, results_dir)
            results[name] = result
            m = result["metrics"]
            print(
                f"  {name:<24} R@10 {m['recall@10']['mean']:.3f}  MRR {m['mrr']['mean']:.3f}"
                f"  nDCG@10 {m['ndcg@10']['mean']:.3f}"
            )
    client.close()  # release the local Qdrant lock cleanly (avoids a shutdown traceback on Windows)

    # Word-based reference rows from the original Phase 3 run (files are read, never rewritten).
    for ref in cfg["reference_results"]:
        result = json.loads((results_dir / f"{ref}.json").read_text(encoding="utf-8"))
        result["reference"] = True
        results[result["name"]] = result

    sizes = [
        config_name(chunking["unit"], s["chunk_size"], s["overlap"]) for s in chunking["compare"]
    ]
    small, large = sizes[0], sizes[-1]
    pairs = [
        (f"dense {large} vs dense {small}", results[f"dense_{large}"], results[f"dense_{small}"]),
        (f"bm25 {large} vs bm25 {small}", results[f"bm25_{large}"], results[f"bm25_{small}"]),
        (f"dense vs bm25 at {small}", results[f"dense_{small}"], results[f"bm25_{small}"]),
        (f"dense vs bm25 at {large}", results[f"dense_{large}"], results[f"bm25_{large}"]),
    ]
    for retriever in ("bm25", "dense"):
        ref = next(
            (
                r
                for r in results.values()
                if r.get("reference") and r["setup"]["retriever"] == retriever
            ),
            None,
        )
        if ref is not None:
            label = f"{retriever} {small} vs {retriever} {ref['name']} (words)"
            pairs.append((label, results[f"{retriever}_{small}"], ref))

    table_path = REPO_ROOT / "reports" / "phase3_tokens_retrieval.md"
    text = (
        "# Phase 3 (token-sized chunks): retrieval on dev\n\n"
        "Generated by `python -m ragpapers.evaluation.run_retrieval` from "
        "reports/results/phase3_tokens/*.json, with word-based reference rows from "
        "reports/results/phase3/.\n\n"
        + make_table(list(results.values()))
        + "\n## Paired comparisons\n\n"
        + comparisons_table(pairs, 2 * cfg["bootstrap_samples"], seed)
    )
    table_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"\nTable written to {table_path}")


if __name__ == "__main__":
    main()

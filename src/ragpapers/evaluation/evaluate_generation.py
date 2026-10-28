"""Generation results per retrieval setup: metrics with 95% intervals, tokens, latency.

Reads, for each setup that has been run (no API call is made):
    reports/results/generation/<split>/<setup>.answers.jsonl            run_generation
    reports/results/generation/<split>/<setup>.judgments.<judge>.jsonl  judge (config `judge`)
    reports/results/judge_validation.json                               agreement

Per question:
    abstention_accuracy   all questions, by rule: abstained iff the question is unanswerable
    faithfulness, citation_correctness, relevance
                          answered questions (abstentions have no claims), judge labels
    correctness           answered answerable questions, judge label vs the reference answer

Each metric is a mean over its questions, with a 95% percentile-bootstrap interval over those
questions. Every judge metric is shown next to the judge's agreement with the user's
20 hand grades: faithfulness and relevance are reliable; citation correctness and correctness
(kappa below the 0.6 threshold) are indicative only.

Writes reports/results/generation/<split>/<setup>.json and reports/generation[_test].md.

Usage (PowerShell, from the repo root):
    python -m ragpapers.evaluation.evaluate_generation              # dev
"""

import argparse
import json
import statistics
from datetime import UTC, datetime

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.freeze import require_final_flag
from ragpapers.evaluation.judge import GROUNDING_METRICS, METRICS, answers_path, judgments_path
from ragpapers.evaluation.scoring import bootstrap_ci, write_result
from ragpapers.generation.runlog import load_done

ALL_METRICS = ("abstention_accuracy", *METRICS)
HEADERS = {
    "abstention_accuracy": "Abstention acc.",
    "faithfulness": "Faithfulness",
    "citation_correctness": "Citation corr.",
    "relevance": "Relevance",
    "correctness": "Correctness",
}
RELIABLE = ("faithfulness", "relevance")  # the user's reporting rule, from the agreement check
SETUP_NAMES = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + reranker",
}


def score_questions(answers: dict, judgments: dict) -> list[dict]:
    """One record per question: its scores (None where a metric does not apply)."""
    out = []
    for qid in sorted(answers):
        a = answers[qid]
        labels = judgments.get(qid, {}).get("labels") or dict.fromkeys(METRICS)
        scores = {"abstention_accuracy": int(a["abstained"] == (a["type"] == "unanswerable"))}
        for metric in GROUNDING_METRICS:
            scores[metric] = None if a["abstained"] else labels.get(metric)
        answered_answerable = not a["abstained"] and a["type"] != "unanswerable"
        scores["correctness"] = labels.get("correctness") if answered_answerable else None
        out.append({"id": qid, "type": a["type"], "abstained": a["abstained"], "scores": scores})
    return out


def summarize(per_question: list[dict], n_samples: int, seed: int) -> dict:
    """{metric: {"mean", "ci_low", "ci_high", "n", "n_missing"}} over applicable questions.

    `n_missing` counts questions the metric applies to but that have no judge label (not yet
    judged, or a judge format error); they are left out of the mean and reported.
    """
    out = {}
    for metric in ALL_METRICS:
        applicable = [q for q in per_question if _applies(q, metric)]
        values = [q["scores"][metric] for q in applicable if q["scores"][metric] is not None]
        entry = {"n": len(values), "n_missing": len(applicable) - len(values)}
        if values:
            low, high = bootstrap_ci(values, n_samples, seed)
            entry |= {"mean": statistics.fmean(values), "ci_low": low, "ci_high": high}
        out[metric] = entry
    return out


def _applies(q: dict, metric: str) -> bool:
    if metric == "abstention_accuracy":
        return True
    if q["abstained"]:
        return False
    return metric != "correctness" or q["type"] != "unanswerable"


def usage(answers: dict) -> dict:
    """Generator tokens and latency per query."""
    records = list(answers.values())
    latencies = [a["latency_ms"]["generation"] for a in records]
    measured = [x for x in latencies if x is not None]
    return {
        "n_queries": len(records),
        "input_tokens_mean": statistics.fmean(a["input_tokens"] for a in records),
        "output_tokens_mean": statistics.fmean(a["output_tokens"] for a in records),
        "reasoning_tokens_mean": statistics.fmean(a["reasoning_tokens"] or 0 for a in records),
        "generation_latency_ms_p50": statistics.median(measured) if measured else None,
        "generation_latency_n": len(measured),  # 0 = not measured (see run_generation)
        "cost_usd": 0.0,  # free tiers only
    }


def check_same_commit(commit: str, *record_sets: dict) -> None:
    """Test results must all come from the frozen commit; refuse to mix states."""
    found = {r.get("commit") for records in record_sets for r in records.values()}
    if found != {commit}:
        raise SystemExit(f"test records come from {sorted(map(str, found))}, not {commit}")


def fmt_metric(m: dict) -> str:
    if "mean" not in m:
        return "not judged" if m["n_missing"] else "-"
    text = f"{m['mean']:.3f} [{m['ci_low']:.2f}, {m['ci_high']:.2f}] n={m['n']}"
    return text + (f" ({m['n_missing']} missing)" if m["n_missing"] else "")


def fmt_agreement(m: dict) -> str:
    kappa = "n/a" if m["kappa"] is None else f"{m['kappa']:.2f}"
    return f"kappa {kappa}, raw {m['raw']:.2f}"


def make_report(results: dict, agreement: dict, meta: dict) -> str:
    header = "| Setup | " + " | ".join(HEADERS[m] for m in ALL_METRICS) + " |"
    rule = "|---|" + "---|" * len(ALL_METRICS)
    lines = [
        f"# Generation on {meta['split']}",
        "",
        "Generated by `python -m ragpapers.evaluation.evaluate_generation`. Generator: "
        f"`{meta['generator']}`, top {meta['context_k']} chunks of each retrieval setup. "
        f"Judge: `{meta['judge']}`. Brackets: 95% bootstrap interval over questions; "
        "n: questions the metric applies to in that setup (abstentions have no judge metrics).",
        "",
        header,
        rule,
    ]
    for setup, result in results.items():
        cells = [fmt_metric(result["metrics"][m]) for m in ALL_METRICS]
        lines.append(f"| {SETUP_NAMES.get(setup, setup)} | " + " | ".join(cells) + " |")
    lines.append(
        "| Judge agreement | by rule | "
        + " | ".join(
            fmt_agreement(agreement[m]) + (" (reliable)" if m in RELIABLE else " (indicative)")
            for m in METRICS
        )
        + " |"
    )
    lines += [
        "",
        f"Judge agreement: the judge's labels vs the user's blind grades of {meta['n_graded']} "
        "hybrid dev answers (Cohen's kappa and raw agreement). Faithfulness and relevance are "
        "reliable; citation correctness and correctness fall below the kappa 0.6 threshold and "
        "are indicative only. Abstention accuracy is scored by rule, not by the judge.",
        "",
        "## Tokens and latency per query (generator)",
        "",
        "| Setup | Input tokens | Output tokens (of which reasoning) | Generation latency p50 |",
        "|---|---|---|---|",
    ]
    for setup, result in results.items():
        u = result["usage"]
        latency = (
            f"{u['generation_latency_ms_p50']:.0f} ms"
            if u["generation_latency_ms_p50"] is not None
            else "not measured"
        )
        lines.append(
            f"| {SETUP_NAMES.get(setup, setup)} | {u['input_tokens_mean']:.0f} | "
            f"{u['output_tokens_mean']:.0f} ({u['reasoning_tokens_mean']:.0f}) | {latency} |"
        )
    lines += [
        "",
        "Cost: $0 (free tiers). Retrieval latency per stage is in reports/retrieval.md. "
        "Generation latency is the API call alone; it is not measured where the stored replies "
        "predate that measurement.",
        "",
        f"Bootstrap: {meta['bootstrap_samples']} resamples of questions, seed {meta['seed']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generation results table (no API calls).")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--final-test-run", action="store_true", help="required for --split test")
    args = parser.parse_args(argv)

    commit = require_final_flag(args.split, args.final_test_run)  # None on dev
    config = load_config()
    seed, n_samples = config["seed"], config["generation_eval"]["bootstrap_samples"]
    judge_model = config["judge"]["model"]
    validation = json.loads(
        (REPO_ROOT / config["paths"]["results_dir"] / "judge_validation.json").read_text("utf-8")
    )
    agreement = validation["metrics"]
    n_graded = next(iter(agreement.values()))["n"]

    results = {}
    created = datetime.now(UTC).isoformat(timespec="seconds")
    for setup in config["generation_eval"]["setups"]:
        answers = load_done(answers_path(config, args.split, setup))
        if not answers:
            continue
        judgments = load_done(judgments_path(config, args.split, setup, judge_model))
        if commit is not None:
            check_same_commit(commit, answers, judgments)
        per_question = score_questions(answers, judgments)
        some = next(iter(answers.values()))
        results[setup] = {
            "name": setup,
            "phase": f"generation/{args.split}",
            "created_at": created,
            "split": args.split,
            "n_questions": len(answers),
            "setup": {
                "retriever": setup,
                "context_k": config["generation"]["context_k"],
                "generator": some["model"],
                "generator_settings": {
                    k: config["llm"][k]
                    for k in ("temperature", "max_output_tokens", "reasoning_effort")
                },
                "judge": judge_model,
                "judge_agreement": agreement,
            },
            "metrics": summarize(per_question, n_samples, seed),
            "usage": usage(answers),
            "per_question": per_question,
        }
        if commit is not None:
            results[setup]["commit"] = commit  # the frozen state of the test run
        write_result(results[setup], REPO_ROOT / config["paths"]["results_dir"])
    if not results:
        raise SystemExit(f"no {args.split} answers yet: run ragpapers.evaluation.run_generation")

    meta = {
        "split": args.split,
        "generator": config["llm"]["model"],
        "context_k": config["generation"]["context_k"],
        "judge": judge_model,
        "n_graded": n_graded,
        "bootstrap_samples": n_samples,
        "seed": seed,
    }
    name = "generation.md" if args.split == "dev" else "generation_test.md"
    report = REPO_ROOT / "reports" / name
    report.write_text(make_report(results, agreement, meta), encoding="utf-8", newline="\n")
    print(
        f"Setups: {', '.join(results)}. Wrote reports/results/generation/{args.split}/*.json "
        f"and {report}"
    )


if __name__ == "__main__":
    main()

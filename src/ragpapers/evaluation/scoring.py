"""Shared scoring for retrieval evaluation: per-question metrics, bootstrap intervals, result files.

For one question, `score_question` credits the retrieved chunks against the question's evidence
groups (evaluation/metrics.py) and returns Recall@k, reciprocal rank and nDCG@k. Means over
questions get 95% percentile-bootstrap intervals (resampling questions), and two setups run on
the same questions are compared with a paired bootstrap of their per-question differences.
"""

import json
import random
import statistics
from pathlib import Path

from ragpapers.evaluation.metrics import credit_quotes, ndcg_at_k, recall_at_k, reciprocal_rank


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


def fmt(metric: dict, with_ci: bool = False) -> str:
    text = f"{metric['mean']:.3f}"
    if with_ci:
        text += f" [{metric['ci_low']:.2f}, {metric['ci_high']:.2f}]"
    return text


def write_result(result: dict, results_dir: Path) -> Path:
    path = results_dir / result["phase"] / f"{result['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path

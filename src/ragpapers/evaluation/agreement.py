"""Judge validation: does the judge grade like the user? Cohen's kappa + raw agreement.

Raw agreement is the share of answers where both give the same label. Cohen's kappa corrects
it for chance: kappa = (p_observed - p_chance) / (1 - p_chance), where p_chance is the
agreement two raters with these label frequencies would reach by luck. 1 = perfect, 0 = no
better than chance.

The rule (fixed before any grading, see configs/default.yaml `judge_validation`):
- per metric, over the sample answers both graded (abstentions have no labels);
- skewed metric: one label is > skew_share of the user's grades. Kappa is unstable there
  (or undefined), so the metric uses raw agreement >= raw_min instead of kappa >= kappa_min;
- the judge is poor if ANY metric fails its test;
- if the primary judge is poor, the fallback judge is scored on the same answers, with the
  same skew decisions (they come from the user's grades), and the judge with the higher mean
  score (kappa, or raw agreement for skewed metrics) is kept.

Usage (PowerShell, from the repo root, after both the grades and the judgments exist):
    python -m ragpapers.evaluation.agreement
"""

import json
import statistics
from collections import Counter

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.judge import judgments_path, load_human_grades, sample_ids
from ragpapers.generation.runlog import load_done


def raw_agreement(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa of two label lists; None when undefined (both raters use one label)."""
    p_observed = raw_agreement(a, b)
    n, count_a, count_b = len(a), Counter(a), Counter(b)
    p_chance = sum(count_a[label] * count_b[label] for label in set(a) | set(b)) / n**2
    if p_chance == 1:
        return None
    return (p_observed - p_chance) / (1 - p_chance)


def metric_agreement(human: list[int], judge: list[int], rule: dict) -> dict:
    """Agreement on one metric, and whether it passes the rule."""
    majority_share = Counter(human).most_common(1)[0][1] / len(human)
    skewed = majority_share > rule["skew_share"]
    kappa, raw = cohen_kappa(human, judge), raw_agreement(human, judge)
    if skewed:
        score, passed = raw, raw >= rule["raw_min"]
    else:
        score, passed = kappa, kappa is not None and kappa >= rule["kappa_min"]
    return {
        "n": len(human),
        "kappa": kappa,
        "raw": raw,
        "human_majority_share": majority_share,
        "skewed": skewed,
        "score": score,  # what the rule compares: raw if skewed, else kappa
        "passed": passed,
    }


def judge_agreement(human: dict[str, dict], judge: dict[str, dict], rule: dict) -> dict:
    """Per-metric agreement of one judge with the user's grades, plus the overall verdict.

    `human` and `judge` map question ID -> {metric: 0/1/None}. Only IDs with a label from
    both are compared; a judge format error (None) on a graded answer counts as disagreement.
    """
    per_metric = {}
    for metric in rule["metrics"]:
        ids = sorted(i for i in human if human[i].get(metric) in (0, 1))
        if not ids:
            raise ValueError(f"no user grades for {metric}")
        h = [human[i][metric] for i in ids]
        j = [judge.get(i, {}).get(metric) for i in ids]
        j = [label if label in (0, 1) else -1 for label in j]  # -1 never matches the user
        per_metric[metric] = metric_agreement(h, j, rule)
    scores = [m["score"] if m["score"] is not None else 0.0 for m in per_metric.values()]
    return {
        "metrics": per_metric,
        "poor": not all(m["passed"] for m in per_metric.values()),
        "mean_score": statistics.fmean(scores),
    }


def choose_judge(primary: dict, fallback: dict | None) -> str:
    """'judge' unless it is poor and the fallback agrees better (higher mean score)."""
    if not primary["poor"]:
        return "judge"
    if fallback is None:
        raise ValueError("the primary judge is poor: score the fallback on the same answers")
    return "judge_fallback" if fallback["mean_score"] > primary["mean_score"] else "judge"


def _labels(path) -> dict[str, dict]:
    return {qid: r["labels"] for qid, r in load_done(path).items()}


def format_report(name: str, result: dict) -> str:
    lines = [
        f"{name}: {'POOR' if result['poor'] else 'ok'} (mean score {result['mean_score']:.3f})"
    ]
    for metric, m in result["metrics"].items():
        kappa = "undefined" if m["kappa"] is None else f"{m['kappa']:.3f}"
        test = "raw >= min (skewed)" if m["skewed"] else "kappa >= min"
        lines.append(
            f"  {metric:21s} n={m['n']:2d}  kappa {kappa:>9s}  raw {m['raw']:.3f}  "
            f"user majority {m['human_majority_share']:.0%}  {test}: "
            f"{'pass' if m['passed'] else 'FAIL'}"
        )
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    jv = config["judge_validation"]
    ids = sample_ids(config)
    human = {i: g for i, g in load_human_grades(REPO_ROOT / jv["human_grades"]).items() if i in ids}
    if len(human) != len(ids):
        raise SystemExit("the user's grades are incomplete")
    rule = {k: jv[k] for k in ("metrics", "kappa_min", "skew_share", "raw_min")}

    results = {}
    for role in ("judge", "judge_fallback"):
        section = config["judge"]
        model = section["model"] if role == "judge" else section["fallback_model"]
        path = judgments_path(config, "dev", jv["setup"], model)
        if path.exists():
            labels = {i: v for i, v in _labels(path).items() if i in ids}
            results[role] = judge_agreement(human, labels, rule) | {"model": model}
            print(format_report(f"{role} ({model})", results[role]))
    if "judge" not in results:
        raise SystemExit("no judgments from the primary judge yet")

    try:
        chosen = choose_judge(results["judge"], results.get("judge_fallback"))
    except ValueError as err:
        print(f"\n{err}")
        chosen = None
    print(f"\nKept judge: {chosen}")
    out = REPO_ROOT / config["paths"]["results_dir"] / "judge_validation.json"
    out.write_text(
        json.dumps({"rule": rule, "results": results, "chosen": chosen}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

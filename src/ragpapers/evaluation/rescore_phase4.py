"""Re-score every Phase 4 setup with the alternative evidence quotes, from the stored rankings.

No retrieval is run: each result file in reports/results/phase4*/ stores, per question, the chunk
IDs it retrieved. Those rankings are scored twice with the same metrics:
    original  one quote per evidence item (as when the experiments were run)
    revised   each evidence item = its quote + approved alternatives (eval/questions.jsonl)
The original scores must reproduce the stored metrics exactly (checked). The earlier decision
rules are then re-applied unchanged to the revised scores:
    rerank depth M   smallest M whose reranked Recall@5 is within `depth_study.tolerance` of the
                     best M (ties: smaller M)
    reranker         best variant by Recall@5 (ties: a, b, c), adopted only if it beats plain
                     hybrid by at least `reranker_followup.min_gain` (point estimate)

Usage (PowerShell, from the repo root):
    python -m ragpapers.evaluation.rescore_phase4
"""

import json
from datetime import UTC, datetime

from ragpapers.chunking.build import chunks_path, load_chunks
from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.candidate_depth import choose_rerank_depth
from ragpapers.evaluation.questions import evidence_groups, load_questions
from ragpapers.evaluation.reranker_followup import adopt, choose_variant
from ragpapers.evaluation.run_phase4 import METRICS
from ragpapers.evaluation.run_retrieval import aggregate, fmt, paired_difference, score_question

PHASE = "phase4_rescored"
SETUPS = {  # label -> (result file, key inside per_question or None)
    "bm25": ("phase4/bm25.json", None),
    "dense": ("phase4/dense.json", None),
    "hybrid": ("phase4/hybrid.json", None),
    "hybrid+rerank (M=20)": ("phase4/hybrid_rerank.json", None),
    "depth M=20": ("phase4_depth/candidate_depth.json", "20"),
    "depth M=50": ("phase4_depth/candidate_depth.json", "50"),
    "depth M=100": ("phase4_depth/candidate_depth.json", "100"),
    "(a) MiniLM": ("phase4_reranker/variant_a.json", None),
    "(b) bge-reranker-base": ("phase4_reranker/variant_b.json", None),
    "(c) RRF(hybrid, b)": ("phase4_reranker/variant_c.json", None),
}
TABLE_ROWS = ("bm25", "dense", "hybrid", "hybrid+rerank (M=20)")
HEADERS = {
    "recall@3": "Recall@3",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "mrr": "MRR",
    "ndcg@10": "nDCG@10",
}


def stored_rankings(result: dict, key: str | None) -> tuple[list[dict], dict]:
    """(per-question entries, stored metrics) of one result file."""
    if key is None:
        return result["per_question"], result["metrics"]
    return result["per_question"][key], result["by_depth"][key]["metrics"]


def rescore(entries: list[dict], questions: dict, by_id: dict, cfg: dict, alternatives: bool):
    """Per-question scores of the stored rankings (top `top_n` of each)."""
    out = []
    for entry in entries:
        q = questions[entry["id"]]
        top = [by_id[i] for i in entry["retrieved"][: cfg["top_n"]]]
        groups = evidence_groups(q, with_alternatives=alternatives)
        scored = score_question(top, groups, cfg["ks"], cfg["quote_match_threshold"])
        out.append({"id": entry["id"], "type": entry["type"], "scores": scored["scores"]})
    return out


def as_result(name: str, per_question: list[dict]) -> dict:
    return {"name": name, "per_question": per_question}


def diff_cell(a: dict, b: dict, metric: str, n_samples: int, seed: int) -> str:
    diff, low, high = paired_difference(a, b, metric, n_samples, seed)
    mark = " *" if low > 0 or high < 0 else ""
    return f"{diff:+.3f} [{low:+.2f}, {high:+.2f}]{mark}"


def main() -> None:
    config = load_config()
    cfg, seed = config["retrieval_eval"], config["seed"]
    n_samples = cfg["bootstrap_samples"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    questions = {q["id"]: q for q in load_questions(REPO_ROOT / config["paths"]["eval_questions"])}
    ret = config["retrieval"]
    chunks = load_chunks(
        chunks_path(config, config["chunking"]["unit"], ret["chunk_size"], ret["overlap"])
    )
    by_id = {c["chunk_id"]: c for c in chunks}

    scores = {}
    for label, (path, key) in SETUPS.items():
        entries, stored = stored_rankings(
            json.loads((results_dir / path).read_text(encoding="utf-8")), key
        )
        original = rescore(entries, questions, by_id, cfg, alternatives=False)
        revised = rescore(entries, questions, by_id, cfg, alternatives=True)
        orig_metrics = aggregate(original, n_samples, seed)
        for metric in METRICS:  # the stored numbers must be reproduced exactly
            if abs(orig_metrics[metric]["mean"] - stored[metric]["mean"]) > 1e-12:
                raise SystemExit(f"{label} {metric}: re-score does not reproduce the stored value")
        scores[label] = {
            "original": as_result(f"{label} original", original),
            "revised": as_result(f"{label} revised", revised),
            "metrics_original": orig_metrics,
            "metrics_revised": aggregate(revised, n_samples, seed),
        }
    print(f"Original scores reproduced exactly for all {len(scores)} setups.")

    def mean(label: str, version: str, metric: str) -> float:
        return scores[label][f"metrics_{version}"][metric]["mean"]

    # --- decision rules, unchanged, on both versions --------------------------------------------
    decisions = {}
    for version in ("original", "revised"):
        depths = {m: mean(f"depth M={m}", version, "recall@5") for m in (20, 50, 100)}
        m_choice = choose_rerank_depth(depths, config["depth_study"]["tolerance"])
        variants = {
            v: mean(lbl, version, "recall@5")
            for v, lbl in (
                ("a", "(a) MiniLM"),
                ("b", "(b) bge-reranker-base"),
                ("c", "(c) RRF(hybrid, b)"),
            )
        }
        base = choose_variant({v: variants[v] for v in ("a", "b")}, ["a", "b"])
        best = choose_variant(variants, ["a", "b", "c"])
        hybrid_r5 = mean("hybrid", version, "recall@5")
        gain = config["reranker_followup"]["min_gain"]
        decisions[version] = {
            "depth_recall5": depths,
            "M": m_choice,
            "variant_recall5": variants,
            "base_for_c": base,
            "best_variant": best,
            "hybrid_recall5": hybrid_r5,
            "adopted": adopt(variants[best], hybrid_r5, gain),
            "adopted_vs_fixed_0.578": adopt(
                variants[best], config["reranker_followup"]["hybrid_recall5"], gain
            ),
        }

    # --- report ----------------------------------------------------------------------------------
    lines = [
        "# Phase 4 re-scored with alternative evidence quotes (dev)",
        "",
        "Generated by `python -m ragpapers.evaluation.rescore_phase4` from the rankings stored in "
        "reports/results/phase4*/ (no retrieval re-run). **Original** = one quote per evidence "
        "item; **revised** = quote + approved alternatives (same paper, same fact). Same 30 "
        "answerable dev questions, same top-10 rankings; the original column reproduces the "
        "stored results exactly.",
        "",
        "## Headline: Phase 4 retrieval table (revised scores)",
        "",
        "These are the project's reported retrieval numbers. Evidence items may list up to 3 "
        "alternative quotes: other verbatim spans of the same paper stating the same fact, chosen "
        "blind to retrieval output (candidates come only from the paper text and the reference "
        "answer) and approved under a strict same-fact rule. An item counts as found once, at the "
        "first chunk containing any of its quotes. 40 alternatives on 28 of 101 evidence items. "
        "The original single-quote scores are kept in the side-by-side table below.",
        "",
        "| Setup | " + " | ".join(HEADERS[m] for m in METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
        *(
            f"| {label} | "
            + " | ".join(fmt(scores[label]["metrics_revised"][k], True) for k in METRICS)
            + " |"
            for label in TABLE_ROWS
        ),
        "",
        "## Phase 4 table: original vs revised",
        "",
        "| Setup | Version | "
        + " | ".join(
            m.replace("recall", "Recall").replace("ndcg", "nDCG").replace("mrr", "MRR")
            for m in METRICS
        )
        + " |",
        "|---|---|" + "---|" * len(METRICS),
    ]
    for label in TABLE_ROWS:
        for version in ("original", "revised"):
            m = scores[label][f"metrics_{version}"]
            lines.append(
                f"| {label} | {version} | " + " | ".join(fmt(m[k], True) for k in METRICS) + " |"
            )
    lines += [
        "",
        "## Revised minus original (paired, same rankings)",
        "",
        "| Setup | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for label in TABLE_ROWS:
        s = scores[label]
        lines.append(
            f"| {label} | "
            + " | ".join(
                diff_cell(s["revised"], s["original"], k, 2 * n_samples, seed) for k in METRICS
            )
            + " |"
        )
    lines += [
        "",
        "## Paired differences against hybrid, original vs revised",
        "",
        "| Setup - hybrid | Version | " + " | ".join(METRICS) + " |",
        "|---|---|" + "---|" * len(METRICS),
    ]
    for label in ("bm25", "dense", "hybrid+rerank (M=20)"):
        for version in ("original", "revised"):
            cells = [
                diff_cell(scores[label][version], scores["hybrid"][version], k, 2 * n_samples, seed)
                for k in METRICS
            ]
            lines.append(f"| {label} | {version} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Paired bootstrap over questions; * = the interval excludes 0.",
        "",
        "## Decision rules re-applied (unchanged)",
        "",
        "| Rule | Original | Revised |",
        "|---|---|---|",
    ]
    o, r = decisions["original"], decisions["revised"]
    tolerance, gain = config["depth_study"]["tolerance"], config["reranker_followup"]["min_gain"]

    def row(label: str, cell) -> str:
        return f"| {label} | {cell(o)} | {cell(r)} |"

    def depths(d: dict) -> str:
        return ", ".join(f"M={m}: {v:.3f}" for m, v in d["depth_recall5"].items())

    def variants(d: dict) -> str:
        return ", ".join(f"{v}: {x:.3f}" for v, x in d["variant_recall5"].items())

    def best(d: dict) -> str:
        delta = d["variant_recall5"][d["best_variant"]] - d["hybrid_recall5"]
        return f"{d['best_variant']} ({delta:+.3f})"

    def adopted(d: dict) -> str:
        return "**yes**" if d["adopted"] else "**no: keep plain hybrid**"

    lines += [
        row("Rerank depth: reranked Recall@5", depths),
        row(f"Chosen M (within {tolerance} of best, ties smaller)", lambda d: f"**{d['M']}**"),
        row("Reranker variants: Recall@5", variants),
        row("Plain hybrid Recall@5", lambda d: f"{d['hybrid_recall5']:.3f}"),
        row("Best variant (gain vs hybrid)", best),
        row(f"Adopt a reranker (needs >= +{gain} vs hybrid, same ruler)", adopted),
        row(
            "Same rule against the fixed 0.578",
            lambda d: "yes" if d["adopted_vs_fixed_0.578"] else "no",
        ),
        "",
        f"Variant (c) was built on {o['base_for_c']} (the better of a/b on the original "
        f"scores); on the revised scores the better of a/b is {r['base_for_c']}.",
        "",
        f"Bootstrap: {n_samples} resamples ({2 * n_samples} for paired differences), seed {seed}.",
    ]
    report = REPO_ROOT / "reports" / "phase4_rescored.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    out = results_dir / PHASE / "rescored.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": PHASE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": cfg["split"],
        "n_questions": len(scores["hybrid"]["revised"]["per_question"]),
        "n_alternatives": sum(
            len(ev.get("alternatives", [])) for q in questions.values() for ev in q["evidence"]
        ),
        "decisions": decisions,
        "setups": {
            label: {
                "metrics_original": s["metrics_original"],
                "metrics_revised": s["metrics_revised"],
                "per_question_revised": s["revised"]["per_question"],
            }
            for label, s in scores.items()
        },
    }
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"M: {o['M']} -> {r['M']}; reranker adopted: {o['adopted']} -> {r['adopted']}")
    print(f"Wrote {report} and {out}")


if __name__ == "__main__":
    main()

"""Phase 4 reranker follow-up: do a stronger reranker, or fusing it with hybrid, beat hybrid?

Fixed for every variant: the `retrieval` chunk set; hybrid = RRF of BM25 top-`fusion_depth` and
dense top-`fusion_depth`; the top `rerank_depth` fused chunks are reranked; the top-n is scored.
    (a) MiniLM cross-encoder, header + text (`dense_text`), replacing the hybrid order of the pool
    (b) bge-reranker-base, same input
    (c) the better of (a)/(b) by Recall@5 (ties: (a)), fused with hybrid via RRF:
        RRF of [the full hybrid ranking, the reranked pool]
The rule (best Recall@5; adopt only if it beats hybrid by `min_gain`) is in configs/default.yaml
(`reranker_followup`) and was fixed before this ran. Paired differences against hybrid are
reported for every variant either way.

It also writes the evidence for a hand check of the questions where (a) lost the most reciprocal
rank against hybrid: (a)'s top chunks, to judge whether they are on topic without the gold quote.

Usage (PowerShell, from the repo root, on an idle machine):
    python -m ragpapers.evaluation.reranker_followup
"""

import statistics
from datetime import UTC, datetime

from ragpapers.chunking.build import dense_text
from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.hardware import hardware_info, hardware_line
from ragpapers.evaluation.questions import evidence_groups, load_questions
from ragpapers.evaluation.run_phase4 import METRICS, evidence_ranks
from ragpapers.evaluation.run_retrieval import (
    aggregate,
    fmt,
    paired_difference,
    score_question,
    write_result,
)
from ragpapers.evaluation.split import load_split
from ragpapers.retrieval.hybrid import reciprocal_rank_fusion
from ragpapers.retrieval.pipeline import first_stage, load_indexes, rerank_top
from ragpapers.retrieval.rerank import Reranker

PHASE = "phase4_reranker"
FLOAT_SLACK = 1e-9


def fuse_with_hybrid(hybrid_ids: list[str], reranked_ids: list[str], rrf_k: int) -> list[str]:
    """Variant (c): RRF of the full hybrid ranking and the reranker's ordering of its pool."""
    return [cid for cid, _ in reciprocal_rank_fusion([hybrid_ids, reranked_ids], rrf_k=rrf_k)]


def choose_variant(recall5: dict[str, float], order: list[str]) -> str:
    """The variant with the best Recall@5; ties go to the earliest in `order`."""
    best = max(recall5[v] for v in order)
    return next(v for v in order if best - recall5[v] <= FLOAT_SLACK)


def adopt(best_value: float, hybrid_value: float, min_gain: float) -> bool:
    """Adopt the best variant only if it beats hybrid by at least `min_gain` (point estimate)."""
    return best_value - hybrid_value >= min_gain - FLOAT_SLACK


def biggest_losses(hybrid_rr: dict[str, float], variant_rr: dict[str, float], n: int) -> list:
    """The n question IDs where the variant lost the most reciprocal rank (ties: by ID)."""
    losses = [(hybrid_rr[q] - variant_rr[q], q) for q in hybrid_rr if hybrid_rr[q] > variant_rr[q]]
    return [q for _, q in sorted(losses, key=lambda item: (-item[0], item[1]))[:n]]


def main() -> None:
    config = load_config()
    seed = config["seed"]
    eval_cfg, ret, fu = config["retrieval_eval"], config["retrieval"], config["reranker_followup"]
    threshold, n_samples, top_n = (
        eval_cfg["quote_match_threshold"],
        eval_cfg["bootstrap_samples"],
        eval_cfg["top_n"],
    )
    depth = ret["rerank_depth"]

    split_ids = load_split(REPO_ROOT / config["paths"]["eval_splits"], eval_cfg["split"])
    dev = [
        q
        for q in load_questions(REPO_ROOT / config["paths"]["eval_questions"])
        if q["id"] in split_ids
    ]
    answerable = [q for q in dev if q["evidence"]]
    print(f"{len(dev)} {eval_cfg['split']} questions, {len(answerable)} answerable")

    indexes = load_indexes(config)
    by_id = indexes["by_id"]
    hybrid = {
        q["id"]: first_stage(q["question"], indexes, ret["fusion_depth"], ret["rrf_k"])[0]["hybrid"]
        for q in dev
    }
    indexes["client"].close()

    # --- rerank with each model (latency on every dev question) --------------------------------
    models = {"a": fu["models"]["a"], "b": fu["models"]["b"]}
    reranked, latency, truncated = {}, {}, {}
    for v, model_name in models.items():
        reranker = Reranker(model_name, config["reranker"]["batch_size"])
        rerank_top(dev[0]["question"], hybrid[dev[0]["id"]], depth, by_id, reranker)  # warm-up
        tokenizer, max_len = reranker.model.tokenizer, reranker.model.max_seq_length
        reranked[v], latency[v], n_over = {}, [], 0
        for q in dev:
            ids, ms = rerank_top(q["question"], hybrid[q["id"]], depth, by_id, reranker)
            reranked[v][q["id"]] = ids
            latency[v].append(ms)
            n_over += sum(
                len(tokenizer(q["question"], dense_text(by_id[i]))["input_ids"]) > max_len
                for i in ids
            )
            print(f"  ({v}) {q['id']}: {ms:.0f} ms")
        truncated[v] = {"max_length": max_len, "share": n_over / (depth * len(dev))}
        del reranker

    # --- score every ranking ---------------------------------------------------------------------
    def score(rankings: dict[str, list[str]]) -> list[dict]:
        out = []
        for q in answerable:
            quotes = evidence_groups(q)
            ids = rankings[q["id"]][:top_n]
            scored = score_question([by_id[i] for i in ids], quotes, eval_cfg["ks"], threshold)
            out.append({"id": q["id"], "type": q["type"], "retrieved": ids, **scored})
        return out

    per_question = {"hybrid": score(hybrid), "a": score(reranked["a"]), "b": score(reranked["b"])}
    metrics = {v: aggregate(pq, n_samples, seed) for v, pq in per_question.items()}
    base = choose_variant({v: metrics[v]["recall@5"]["mean"] for v in ("a", "b")}, ["a", "b"])
    fused = {
        qid: fuse_with_hybrid(hybrid[qid], reranked[base][qid], ret["rrf_k"]) for qid in hybrid
    }
    per_question["c"] = score(fused)
    metrics["c"] = aggregate(per_question["c"], n_samples, seed)
    latency["c"] = latency[base]

    hybrid_r5 = metrics["hybrid"]["recall@5"]["mean"]
    if abs(hybrid_r5 - fu["hybrid_recall5"]) > 0.0005:
        raise SystemExit(f"hybrid Recall@5 {hybrid_r5:.3f} differs from the rule's baseline")
    variants = ["a", "b", "c"]
    best = choose_variant({v: metrics[v]["recall@5"]["mean"] for v in variants}, variants)
    # The rule is stated against the Phase 4 table value (0.578): compare with exactly that.
    adopted = adopt(metrics[best]["recall@5"]["mean"], fu["hybrid_recall5"], fu["min_gain"])

    # --- results files -------------------------------------------------------------------------
    hardware = hardware_info()
    created = datetime.now(UTC).isoformat(timespec="seconds")
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    labels = {
        "a": f"(a) {models['a']}",
        "b": f"(b) {models['b']}",
        "c": f"(c) RRF(hybrid, {base})",
    }
    results = {"hybrid": {"name": "hybrid", "per_question": per_question["hybrid"]}}
    for v in variants:
        results[v] = {
            "name": f"variant_{v}",
            "phase": PHASE,
            "created_at": created,
            "split": eval_cfg["split"],
            "n_questions": len(answerable),
            "setup": {
                "variant": labels[v],
                "chunk_config": indexes["name"],
                "fusion_depth": ret["fusion_depth"],
                "rrf_k": ret["rrf_k"],
                "rerank_depth": depth,
                "reranker_model": models[base if v == "c" else v],
                "reranker_input": "dense_text (short header + chunk text)",
                "fused_with_hybrid": v == "c",
                "top_n": top_n,
                "quote_match_threshold": threshold,
            },
            "metrics": metrics[v],
            "latency_ms": {
                "rerank_p50": statistics.median(latency[v]),
                "rerank_mean": statistics.fmean(latency[v]),
                "n_questions": len(dev),
            },
            "truncated": truncated[base if v == "c" else v],
            "hardware": hardware,
            "per_question": per_question[v],
        }
        write_result(results[v], results_dir)

    # --- report --------------------------------------------------------------------------------
    hybrid_m = metrics["hybrid"]
    lines = [
        "# Phase 4: reranker follow-up (dev)",
        "",
        "Generated by `python -m ragpapers.evaluation.reranker_followup`. Same first stage for "
        f"every row: chunks {indexes['name']}, hybrid = RRF (k={ret['rrf_k']}) of BM25 "
        f"top-{ret['fusion_depth']} and dense top-{ret['fusion_depth']}; rerankers rescore the "
        f"top {depth} fused chunks from `dense_text` (short title > section heading + chunk "
        f"text). Top {top_n} scored on {len(answerable)} answerable dev questions; latency on "
        f"all {len(dev)}.",
        "",
        f"Hardware: {hardware_line(hardware)}.",
        "",
        "| Setup | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 | Rerank p50 (ms) "
        "| Pairs truncated |",
        "|---|---|---|---|---|---|---|---|",
        "| hybrid | " + " | ".join(fmt(hybrid_m[k], True) for k in METRICS) + " | - | - |",
    ]
    for v in variants:
        t = truncated[base if v == "c" else v]
        lines.append(
            f"| {labels[v]} | "
            + " | ".join(fmt(metrics[v][k], True) for k in METRICS)
            + f" | {statistics.median(latency[v]):.0f} | {100 * t['share']:.1f}% |"
        )
    lines += [
        "",
        "## Paired differences against hybrid (variant - hybrid)",
        "",
        "| Variant | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|---|---|---|---|---|---|",
    ]
    for v in variants:
        cells = []
        for metric in METRICS:
            diff, low, high = paired_difference(
                results[v], results["hybrid"], metric, 2 * n_samples, seed
            )
            mark = " *" if low > 0 or high < 0 else ""
            cells.append(f"{diff:+.3f} [{low:+.2f}, {high:+.2f}]{mark}")
        lines.append(f"| {labels[v]} | " + " | ".join(cells) + " |")
    best_r5 = metrics[best]["recall@5"]["mean"]
    lines += [
        "",
        "Paired bootstrap over the same questions; * = the interval excludes 0. (c) uses the "
        "rerank latency of its base reranker (the extra fusion takes < 1 ms).",
        "",
        "## Decision",
        "",
        "Rule (fixed before running): choose the variant with the best Recall@5 (ties: a, b, c); "
        f"adopt it only if it beats hybrid's {fu['hybrid_recall5']:.3f} by at least "
        f"{fu['min_gain']} on the point estimate, otherwise keep plain hybrid.",
        "",
        f"Best variant: **{labels[best]}**, Recall@5 {best_r5:.3f} "
        f"({best_r5 - hybrid_r5:+.3f} vs hybrid). "
        + (
            "**Adopted.**"
            if adopted
            else "**Not adopted: plain hybrid is kept** "
            f"(needed >= {fu['hybrid_recall5'] + fu['min_gain']:.3f})."
        ),
        "",
        f"Bootstrap: {n_samples} resamples of questions, seed {seed}.",
    ]
    report = REPO_ROOT / "reports" / "phase4_reranker_followup.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # --- hand-check evidence (for a human to judge; never used to choose) ----------------------
    rr = {v: {p["id"]: p["scores"]["rr"] for p in per_question[v]} for v in ("hybrid", "a")}
    losers = biggest_losses(rr["hybrid"], rr["a"], fu["hand_check_n"])
    by_q = {q["id"]: q for q in answerable}
    check = [
        "# Hand check: questions where reranker (a) lost the most reciprocal rank",
        "",
        "Evidence for a human judgement; not used to choose anything. "
        "For each question: where the gold evidence ranks, and (a)'s top 3 chunks.",
    ]
    for qid in losers:
        q = by_q[qid]
        quotes = evidence_groups(q)
        h_ranks = evidence_ranks([by_id[i]["text"] for i in hybrid[qid]], quotes, threshold)
        a_ranks = evidence_ranks([by_id[i]["text"] for i in reranked["a"][qid]], quotes, threshold)
        check += [
            "",
            f"## {qid} ({q['type']}): RR hybrid {rr['hybrid'][qid]:.3f} -> (a) {rr['a'][qid]:.3f}",
            "",
            f"**Question:** {q['question']}",
            "",
        ]
        for (quote, *_), e, hr, ar in zip(quotes, q["evidence"], h_ranks, a_ranks, strict=True):
            check.append(
                f'- Gold ({e["paper_id"]}): "{quote}" | hybrid rank {hr or "-"}, '
                f"(a) rank {ar or '-'}"
            )
        for rank, cid in enumerate(reranked["a"][qid][:3], start=1):
            c = by_id[cid]
            text = " ".join(c["text"].split())
            check += [
                "",
                f"**(a) #{rank}** `{cid}` {c['title']} > {c['section']}",
                "",
                f"> {text[:700]}{' ...' if len(text) > 700 else ''}",
            ]
    hand = REPO_ROOT / "reports" / "phase4_reranker_handcheck.md"
    hand.write_text("\n".join(check) + "\n", encoding="utf-8", newline="\n")
    print(f"\nBest {labels[best]} R@5 {best_r5:.3f}; adopted: {adopted}\nWrote {report}, {hand}")


if __name__ == "__main__":
    main()

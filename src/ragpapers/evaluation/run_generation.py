"""Generate answers for every question of a split with one retrieval setup (resumable).

    bm25            BM25 top-k
    dense           dense top-k
    hybrid          RRF of BM25 + dense, top-k (the final system)
    hybrid_rerank   the top `rerank_depth` fused chunks reranked by the cross-encoder, top-k

k = `generation.context_k`. Each finished question is appended at once to
reports/results/generation/<split>/<setup>.answers.jsonl; a stopped run (daily quota, crash)
resumes from there, and every LLM reply is cached, so nothing is paid for twice.

Usage (PowerShell, from the repo root):
    python -m ragpapers.evaluation.run_generation --setup hybrid          # estimate only
    python -m ragpapers.evaluation.run_generation --setup hybrid --run    # make the calls

The test split is held out: it also needs --final-test-run.
"""

import argparse

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.judge import answers_path
from ragpapers.evaluation.questions import load_questions
from ragpapers.evaluation.split import load_split
from ragpapers.generation.citations import cite_answer
from ragpapers.generation.client import DailyQuotaReached, make_client
from ragpapers.generation.prompt import build_messages
from ragpapers.generation.runlog import append_record, load_done
from ragpapers.retrieval.pipeline import first_stage, load_indexes, rerank_top


def retrieve_for_setup(question: str, setup: str, indexes: dict, config: dict, reranker=None):
    """The top `context_k` chunk IDs of one setup, and the stage latencies in ms."""
    ret, k = config["retrieval"], config["generation"]["context_k"]
    rankings, ms = first_stage(question, indexes, ret["fusion_depth"], ret["rrf_k"])
    if setup in ("bm25", "dense"):
        ids, ms = rankings[setup], {setup: ms[setup]}
    elif setup == "hybrid":
        ids = rankings["hybrid"]
    elif setup == "hybrid_rerank":
        ids, ms["rerank"] = rerank_top(
            question, rankings["hybrid"], ret["rerank_depth"], indexes["by_id"], reranker
        )
    else:
        raise ValueError(f"unknown setup {setup!r}")
    return ids[:k], ms


def answer_record(q: dict, setup: str, chunk_ids: list[str], ms: dict, response, cited) -> dict:
    return {
        "id": q["id"],
        "type": q["type"],
        "setup": setup,
        "chunk_ids": chunk_ids,  # chunk n of the prompt is chunk_ids[n - 1]
        "text": response.text,
        "abstained": cited.abstained,
        "cited": [s.number for s in cited.sources],
        "invalid_citations": cited.invalid_citations,
        "provider": response.provider,
        "requested_model": response.requested_model,
        "model": response.model,
        "system_fingerprint": response.system_fingerprint,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "finish_reason": response.finish_reason,
        "cached": response.cached,
        "request_hash": response.request_hash,
        # Cache entries written before `wait_ms` existed timed the throttle wait too: their
        # generation latency is not a measurement of the API, so it is left out (None).
        "latency_ms": {
            **ms,
            "generation": response.latency_ms if response.wait_ms is not None else None,
        },
        "wait_ms": response.wait_ms,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate answers for one retrieval setup.")
    parser.add_argument("--setup", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--run", action="store_true", help="make the calls (default: estimate)")
    parser.add_argument("--final-test-run", action="store_true", help="required for --split test")
    args = parser.parse_args(argv)

    config = load_config()
    if args.setup not in config["generation_eval"]["setups"]:
        raise SystemExit(f"unknown setup {args.setup!r}: {config['generation_eval']['setups']}")
    if args.split == "test" and not args.final_test_run:
        raise SystemExit("the test split is held out: add --final-test-run for the final run")

    split_ids = load_split(REPO_ROOT / config["paths"]["eval_splits"], args.split)
    questions = [
        q
        for q in load_questions(REPO_ROOT / config["paths"]["eval_questions"])
        if q["id"] in split_ids
    ]
    out = answers_path(config, args.split, args.setup)
    done = load_done(out)
    todo = [q for q in questions if q["id"] not in done]
    client = make_client(config, "generator")
    reranker = None
    if args.setup == "hybrid_rerank":
        from ragpapers.retrieval.rerank import Reranker

        reranker = Reranker(config["reranker"]["model"], config["reranker"]["batch_size"])

    indexes = load_indexes(config)
    abstain = config["abstention_text"]
    try:
        # Retrieval first (offline), so the estimate uses the real prompts.
        planned = []
        for q in todo:
            ids, ms = retrieve_for_setup(q["question"], args.setup, indexes, config, reranker)
            messages = build_messages(q["question"], [indexes["by_id"][i] for i in ids], abstain)
            planned.append((q, ids, ms, messages))
    finally:
        indexes["client"].close()

    uncached = [p for p in planned if not client.is_cached(p[3])]
    chars = sum(len(m.content) for p in uncached for m in p[3])
    max_out = client.defaults["max_output_tokens"]
    print(
        f"{args.setup} ({args.split}): {len(questions)} questions, {len(done)} already done, "
        f"{len(uncached)} API calls to {client.model}: "
        f"~{chars / config['generation_eval']['chars_per_token'] / 1000:.0f}K input tokens "
        f"+ up to {len(uncached) * max_out / 1000:.0f}K output"
    )
    if not args.run:
        print("Estimate only. Add --run to make the calls.")
        return

    by_id = {c["chunk_id"]: c for c in indexes["chunks"]}
    try:
        for q, ids, ms, messages in planned:
            response = client.generate(messages)
            cited = cite_answer(response.text, [by_id[i] for i in ids], abstain)
            append_record(out, answer_record(q, args.setup, ids, ms, response, cited))
            print(
                f"  {q['id']}: {'abstained' if cited.abstained else 'answered'}"
                f"{' (cached)' if response.cached else ''}"
            )
    except DailyQuotaReached as err:
        raise SystemExit(f"Stopped: {err}") from err
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

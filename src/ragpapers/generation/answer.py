"""Answer one question end to end: retrieve -> prompt -> generate -> cite.

retrieval   plain hybrid (RRF of BM25 + dense), top `generation.context_k` chunks
            (+ the cross-encoder if `retrieval.use_reranker` is true)
prompt      generation/prompt.py: rules + numbered chunks + question
generate    the configured generator (cached: asking again makes no API call)
cite        generation/citations.py: [n] -> paper title + section + link
"""

import argparse
import time

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.questions import load_questions
from ragpapers.evaluation.split import load_split
from ragpapers.generation.citations import cite_answer, format_sources
from ragpapers.generation.client import make_client
from ragpapers.generation.prompt import build_messages
from ragpapers.retrieval.pipeline import elapsed_ms, first_stage, load_indexes, rerank_top


def retrieve(question: str, indexes: dict, config: dict, reranker=None) -> tuple[list, dict]:
    """The top `context_k` chunks for the question, and the retrieval latencies in ms."""
    ret, k = config["retrieval"], config["generation"]["context_k"]
    rankings, ms = first_stage(question, indexes, ret["fusion_depth"], ret["rrf_k"])
    ids = rankings["hybrid"]
    if reranker is not None:
        ids, ms["rerank"] = rerank_top(
            question, ids, ret["rerank_depth"], indexes["by_id"], reranker
        )
    return [indexes["by_id"][i] for i in ids[:k]], ms


def answer_question(question: str, indexes: dict, client, config: dict, reranker=None) -> dict:
    """Retrieved chunks, the model response, the cited answer and the latencies of one question."""
    chunks, ms = retrieve(question, indexes, config, reranker)
    start = time.perf_counter()
    response = client.generate(build_messages(question, chunks, config["abstention_text"]))
    ms["generation_call"] = elapsed_ms(start)  # ~0 when served from the cache
    return {
        "question": question,
        "chunks": chunks,
        "response": response,
        "cited": cite_answer(response.text, chunks, config["abstention_text"]),
        "latency_ms": ms,
    }


def dev_question(question_id: str, config: dict) -> str:
    """The text of a dev-split question; refuses any other split."""
    paths = config["paths"]
    if question_id not in load_split(REPO_ROOT / paths["eval_splits"], "dev"):
        raise SystemExit(f"{question_id} is not a dev question (the test split is held out)")
    for q in load_questions(REPO_ROOT / paths["eval_questions"]):
        if q["id"] == question_id:
            return q["question"]
    raise SystemExit(f"{question_id} not found")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("question", nargs="?", help="the question to answer")
    group.add_argument("--id", help="a dev question ID from eval/questions.jsonl, e.g. q101")
    args = parser.parse_args(argv)

    config = load_config()
    question = dev_question(args.id, config) if args.id else args.question
    client = make_client(config, "generator")
    reranker = None
    if config["retrieval"]["use_reranker"]:
        from ragpapers.retrieval.rerank import Reranker

        reranker = Reranker(config["reranker"]["model"], config["reranker"]["batch_size"])

    indexes = load_indexes(config)
    try:
        result = answer_question(question, indexes, client, config, reranker)
    finally:
        indexes["client"].close()  # releases the local Qdrant lock

    r, cited = result["response"], result["cited"]
    print(f"Question: {question}\n")
    print(cited.text)
    if cited.sources:
        print("\nSources:\n" + format_sources(cited.sources))
    if cited.invalid_citations:
        print(f"\nWarning: cites chunk numbers never shown: {cited.invalid_citations}")
    print(f"\nRetrieved: {', '.join(c['chunk_id'] for c in result['chunks'])}")
    print(f"Model: {r.provider} {r.model} (requested {r.requested_model})")
    print(
        f"Tokens: in {r.input_tokens}, out {r.output_tokens} (reasoning {r.reasoning_tokens}); "
        f"finish {r.finish_reason}; cached {r.cached}"
    )
    print(
        "Latency: "
        + ", ".join(f"{s} {v:.0f} ms" for s, v in result["latency_ms"].items())
        + f"; generation {r.latency_ms:.0f} ms"
    )


if __name__ == "__main__":
    main()

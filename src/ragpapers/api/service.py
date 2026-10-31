"""The served RAG system: the frozen final configuration behind one `ask` call.

    service = RagService.from_config(load_config())   # loads the indexes and generator once
    result = service.ask("Which weights does LoRA freeze?")

The served system is exactly the one evaluated on the held-out test set: plain hybrid
retrieval (RRF of BM25 + dense), the top 8 chunks, gpt-oss-120b on Groq at temperature 0,
and the frozen prompt and citation parser (generation/answer.py runs all of it).
`check_frozen_config` refuses to serve any other configuration, and `ask` takes no
overrides. No reranker and no judge are loaded: the judge is for evaluation only.

Requests are answered one at a time (one lock): the local Qdrant client and the per-minute
throttle are shared, and a laptop demo does not need parallel answers.
"""

import os
import threading
import time
import uuid
from datetime import UTC, datetime

import requests

from ragpapers.api.querylog import QueryLog
from ragpapers.config import REPO_ROOT
from ragpapers.generation.answer import answer_question
from ragpapers.generation.client import DailyQuotaReached, make_client
from ragpapers.generation.providers import ProviderError, RateLimited, TransientError
from ragpapers.retrieval.pipeline import elapsed_ms, load_indexes

# Every setting that changes what the served system retrieves or generates, with the value
# evaluated on the test set (configs/default.yaml at commit EVALUATED_COMMIT).
FROZEN = {
    ("chunking", "chunk_size"): 448,
    ("chunking", "overlap"): 64,
    ("embedding", "model"): "BAAI/bge-small-en-v1.5",
    ("bm25", "k1"): 1.5,
    ("bm25", "b"): 0.75,
    ("retrieval", "rrf_k"): 60,
    ("retrieval", "fusion_depth"): 100,
    ("retrieval", "use_reranker"): False,
    ("generation", "context_k"): 8,
    ("llm", "provider"): "groq",
    ("llm", "model"): "openai/gpt-oss-120b",
    ("llm", "temperature"): 0,
    ("llm", "reasoning_effort"): "low",
    ("llm", "max_output_tokens"): 1024,
    ("abstention_text",): "I couldn't find sufficient evidence in the available sources.",
}
EVALUATED_COMMIT = "99d8e99"
SETUP = "hybrid"  # the retrieval setup name used in the evaluation reports

_MISSING = object()


class FrozenConfigError(ValueError):
    """The configuration differs from the evaluated one, so it must not be served."""


class ServiceUnavailable(Exception):
    """The service cannot answer right now (HTTP 503): no API key, or the daily quota."""


class GeneratorFailed(Exception):
    """The generator API failed after its retries (HTTP 502)."""


def _get(config: dict, path: tuple):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def check_frozen_config(config: dict) -> None:
    """Raise FrozenConfigError listing every setting that differs from FROZEN."""
    problems = []
    for path, expected in FROZEN.items():
        value = _get(config, path)
        # bool is checked apart, since False == 0 in Python
        if value != expected or isinstance(value, bool) != isinstance(expected, bool):
            shown = "missing" if value is _MISSING else repr(value)
            problems.append(f"{'.'.join(path)} is {shown}, the evaluated system uses {expected!r}")
    if problems:
        raise FrozenConfigError(
            f"refusing to serve a configuration that was not evaluated "
            f"(commit {EVALUATED_COMMIT}): " + "; ".join(problems)
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def retrieved_entries(chunks: list[dict], rankings: dict) -> list[dict]:
    """The chunks shown to the generator, best first, with why each one was retrieved.

    rrf_score is the chunk's fused score; bm25_rank / dense_rank are its 1-based ranks in
    each retriever's list (None when that retriever did not return it).
    """
    bm25 = {cid: rank for rank, cid in enumerate(rankings.get("bm25", []), start=1)}
    dense = {cid: rank for rank, cid in enumerate(rankings.get("dense", []), start=1)}
    scores = rankings.get("hybrid_scores", {})
    return [
        {
            "rank": rank,
            "chunk_id": c["chunk_id"],
            "paper_id": c["paper_id"],
            "title": c["title"],
            "year": c["year"],
            "section": c["section"],
            "url": c["section_url"],
            "text": c["text"],
            "rrf_score": scores.get(c["chunk_id"]),
            "bm25_rank": bm25.get(c["chunk_id"]),
            "dense_rank": dense.get(c["chunk_id"]),
        }
        for rank, c in enumerate(chunks, start=1)
    ]


class RagService:
    def __init__(
        self,
        config: dict,
        indexes: dict,
        client,
        query_log: QueryLog,
        client_error: str | None = None,
    ):
        """`client` is the generator's LLMClient, or None with `client_error` saying why."""
        check_frozen_config(config)
        self.config, self.indexes, self.client = config, indexes, client
        self.client_error = client_error
        self.log = query_log
        self.lock = threading.Lock()

    @classmethod
    def from_config(cls, config: dict) -> "RagService":
        """Check the config, then load the indexes and the generator (once, at startup).

        A missing API key does not stop the service: /health reports it and /ask answers 503.
        """
        check_frozen_config(config)  # before spending seconds loading anything
        indexes = load_indexes(config)
        try:
            client, error = make_client(config, "generator"), None
        except RuntimeError as err:  # the API key is not set
            client, error = None, str(err)
        log = QueryLog(REPO_ROOT / config["paths"]["query_log"])
        return cls(config, indexes, client, log, error)

    def close(self) -> None:
        """Release the local Qdrant lock."""
        qdrant = self.indexes.get("client")
        if qdrant is not None:
            qdrant.close()

    def health(self) -> dict:
        """What is loaded and what is served. Cheap: never calls the LLM."""
        llm, ret = self.config["llm"], self.config["retrieval"]
        key_env = self.config["providers"][llm["provider"]].get("api_key_env")
        return {
            "status": "ok" if self.client is not None else "degraded",
            "detail": self.client_error,
            "evaluated_commit": EVALUATED_COMMIT,
            "setup": SETUP,
            "rrf_k": ret["rrf_k"],
            "context_k": self.config["generation"]["context_k"],
            "reranker": None,
            "embedding_model": self.config["embedding"]["model"],
            "generator": {"provider": llm["provider"], "model": llm["model"]},
            "temperature": llm["temperature"],
            "chunk_config": self.indexes.get("name"),
            "n_chunks": len(self.indexes["by_id"]),
            "api_key_set": bool(key_env and os.environ.get(key_env)),  # never the key itself
        }

    def ask(self, question: str) -> dict:
        """Answer one question with the frozen system; every request is logged.

        Raises ServiceUnavailable (no API key, daily quota) or GeneratorFailed (API errors
        after the retries); both are logged with their error before being raised.
        """
        record = {"timestamp": _now(), "request_id": uuid.uuid4().hex, "question": question}
        start = time.perf_counter()
        try:
            if self.client is None:
                raise ServiceUnavailable(self.client_error or "the generator is not configured")
            with self.lock:
                result = answer_question(question, self.indexes, self.client, self.config)
        except DailyQuotaReached as err:
            self._log_failure(record, start, err)
            raise ServiceUnavailable(
                "The generator's daily free-tier quota is used up; try again tomorrow."
            ) from err
        except (ProviderError, RateLimited, TransientError, requests.RequestException) as err:
            self._log_failure(record, start, err)
            raise GeneratorFailed(f"The generator API failed: {err}") from err
        except ServiceUnavailable as err:
            self._log_failure(record, start, err)
            raise

        total = elapsed_ms(start)
        response, cited = result["response"], result["cited"]
        ms = dict(result["latency_ms"])
        # generation: this request's wall time for the call (~0 when served from the cache)
        generation = ms.pop("generation_call")
        latency = {**ms, "generation": generation, "total": total}
        retrieved = retrieved_entries(result["chunks"], result["rankings"])

        self.log.append(
            {
                **record,
                "setup": SETUP,
                "retrieved": [
                    {k: r[k] for k in ("chunk_id", "rrf_score", "bm25_rank", "dense_rank")}
                    for r in retrieved
                ],
                "abstained": cited.abstained,
                "cited": [s.number for s in cited.sources],
                "cited_chunk_ids": [s.chunk_id for s in cited.sources],
                "invalid_citations": cited.invalid_citations,
                "latency_ms": latency,
                "generation_api_ms": response.latency_ms,  # the API call alone (cache: original)
                "wait_ms": response.wait_ms,
                "provider": response.provider,
                "model": response.model,
                "cached": response.cached,
                "request_hash": response.request_hash,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "reasoning_tokens": response.reasoning_tokens,
                "finish_reason": response.finish_reason,
                "error": None,
            }
        )
        return {
            "request_id": record["request_id"],
            "question": question,
            "answer": cited.text,
            "abstained": cited.abstained,
            "citations": [
                {
                    "number": s.number,
                    "chunk_id": s.chunk_id,
                    "paper_id": s.paper_id,
                    "title": s.title,
                    "year": s.year,
                    "section": s.section,
                    "url": s.url,
                }
                for s in cited.sources
            ],
            "invalid_citations": cited.invalid_citations,
            "retrieved": retrieved,
            "latency_ms": latency,
            "model": response.model,
            "cached": response.cached,
        }

    def _log_failure(self, record: dict, start: float, err: Exception) -> None:
        self.log.append(
            {
                **record,
                "setup": SETUP,
                "latency_ms": {"total": elapsed_ms(start)},
                "error": f"{type(err).__name__}: {err}",
            }
        )

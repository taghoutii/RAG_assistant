import copy
import json

import pytest
from fastapi.testclient import TestClient

from ragpapers.api import service as service_module
from ragpapers.api.app import create_app
from ragpapers.api.querylog import QueryLog, read_log
from ragpapers.api.service import (
    WARM_UP_QUERY,
    FrozenConfigError,
    RagService,
    check_frozen_config,
)
from ragpapers.config import load_config
from ragpapers.generation import answer
from ragpapers.generation.client import LLMClient
from ragpapers.generation.providers import FakeProvider, ProviderError, RateLimited

CONFIG = load_config()
ABSTAIN = CONFIG["abstention_text"]
K = CONFIG["generation"]["context_k"]


def chunk(n):
    return {
        "chunk_id": f"p:{n:04d}",
        "paper_id": "p",
        "title": "Paper",
        "year": 2020,
        "section": f"Section {n}",
        "section_url": f"https://arxiv.org/html/p#S{n}",
        "text": f"text {n}",
    }


@pytest.fixture
def fake_retrieval(monkeypatch):
    """20 chunks; hybrid order p:0020 ... p:0001; BM25 returns only the even ones."""
    by_id = {f"p:{n:04d}": chunk(n) for n in range(1, 21)}
    hybrid = list(reversed(by_id))

    def fake_first_stage(question, indexes, fusion_depth, rrf_k):
        rankings = {
            "bm25": [i for i in hybrid if int(i[-4:]) % 2 == 0],
            "dense": hybrid,
            "hybrid": hybrid,
            "hybrid_scores": {i: 1 / (rrf_k + rank) for rank, i in enumerate(hybrid, start=1)},
        }
        return rankings, {"bm25": 1.0, "dense": 2.0, "fusion": 0.1}

    monkeypatch.setattr(answer, "first_stage", fake_first_stage)
    return {"by_id": by_id, "name": "fake"}


def make_service(indexes, tmp_path, script=(), client_error=None):
    fake = FakeProvider(list(script))
    client = None if client_error else LLMClient(fake, CONFIG["llm"]["model"])
    log = QueryLog(tmp_path / "logs" / "queries.jsonl")
    return RagService(CONFIG, indexes, client, log, client_error), fake


def api(service):
    return TestClient(create_app(lambda: service))


# --- /health -----------------------------------------------------------------------------------


def test_health_reports_the_frozen_system_without_calling_the_llm(fake_retrieval, tmp_path):
    service, fake = make_service(fake_retrieval, tmp_path)
    with api(service) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["setup"] == "hybrid" and body["reranker"] is None
    assert (body["rrf_k"], body["context_k"], body["temperature"]) == (60, 8, 0)
    assert body["generator"] == {"provider": "groq", "model": "openai/gpt-oss-120b"}
    assert body["evaluated_commit"] == "99d8e99"
    assert body["n_chunks"] == 20
    assert isinstance(body["api_key_set"], bool)
    assert fake.requests == []


def test_health_never_shows_the_key(fake_retrieval, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-test-key")
    service, _ = make_service(fake_retrieval, tmp_path)
    with api(service) as client:
        r = client.get("/health")
    assert r.json()["api_key_set"] is True
    assert "secret-test-key" not in r.text


def test_health_is_degraded_without_a_generator(fake_retrieval, tmp_path):
    service, _ = make_service(fake_retrieval, tmp_path, client_error="GROQ_API_KEY is not set")
    with api(service) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert "GROQ_API_KEY" in body["detail"]


# --- /ask --------------------------------------------------------------------------------------


def test_ask_returns_answer_citations_retrieved_chunks_and_latencies(fake_retrieval, tmp_path):
    service, fake = make_service(fake_retrieval, tmp_path, ["Fact [2], and more [1]."])
    with api(service) as client:
        r = client.post("/ask", json={"question": "  What is it?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "What is it?"  # surrounding whitespace stripped
    assert body["answer"] == "Fact [2], and more [1]."
    assert body["abstained"] is False
    assert [c["number"] for c in body["citations"]] == [2, 1]  # first-cited order
    assert body["citations"][0]["chunk_id"] == "p:0019"  # [2] = the second chunk shown
    assert body["citations"][0]["url"] == "https://arxiv.org/html/p#S19"

    retrieved = body["retrieved"]
    assert [c["chunk_id"] for c in retrieved] == [f"p:{n:04d}" for n in range(20, 20 - K, -1)]
    first, second = retrieved[0], retrieved[1]
    assert first["rank"] == 1 and first["text"] == "text 20"
    assert first["rrf_score"] == pytest.approx(1 / 61)
    assert (first["bm25_rank"], first["dense_rank"]) == (1, 1)
    assert second["bm25_rank"] is None  # p:0019 is odd: not in the BM25 list

    assert set(body["latency_ms"]) == {"bm25", "dense", "fusion", "generation", "total"}
    assert body["model"] == "openai/gpt-oss-120b" and body["cached"] is False
    (request,) = fake.requests
    assert request.model == "openai/gpt-oss-120b" and request.temperature == 0


def test_an_abstention_has_no_citations(fake_retrieval, tmp_path):
    service, _ = make_service(fake_retrieval, tmp_path, [ABSTAIN])
    with api(service) as client:
        body = client.post("/ask", json={"question": "Unanswerable?"}).json()
    assert body["abstained"] is True
    assert body["answer"] == ABSTAIN
    assert body["citations"] == []
    assert len(body["retrieved"]) == K  # still returned, for debugging


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {},
        {"question": "Q?", "k": 3},  # no overrides of the frozen system
        {"question": "Q?", "model": "other"},
    ],
)
def test_invalid_requests_are_rejected_before_the_pipeline(fake_retrieval, tmp_path, payload):
    service, fake = make_service(fake_retrieval, tmp_path)
    with api(service) as client:
        r = client.post("/ask", json=payload)
    assert r.status_code == 422
    assert fake.requests == []
    assert read_log(service.log.path) == []


def test_daily_quota_gives_503_and_is_logged(fake_retrieval, tmp_path):
    service, _ = make_service(fake_retrieval, tmp_path, [RateLimited("quota", daily=True)])
    with api(service) as client:
        r = client.post("/ask", json={"question": "Q?"})
    assert r.status_code == 503
    assert "quota" in r.json()["detail"]
    (record,) = read_log(service.log.path)
    assert record["error"].startswith("DailyQuotaReached")
    assert record["question"] == "Q?"


def test_a_provider_error_gives_502_and_is_logged(fake_retrieval, tmp_path):
    service, _ = make_service(fake_retrieval, tmp_path, [ProviderError("HTTP 400")])
    with api(service) as client:
        r = client.post("/ask", json={"question": "Q?"})
    assert r.status_code == 502
    (record,) = read_log(service.log.path)
    assert record["error"].startswith("ProviderError")


def test_no_generator_gives_503(fake_retrieval, tmp_path):
    service, _ = make_service(fake_retrieval, tmp_path, client_error="GROQ_API_KEY is not set")
    with api(service) as client:
        r = client.post("/ask", json={"question": "Q?"})
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]
    assert read_log(service.log.path)[0]["error"].startswith("ServiceUnavailable")


# --- startup warm-up ---------------------------------------------------------------------------


@pytest.fixture
def retrieval_calls(fake_retrieval, monkeypatch):
    """The queries first_stage receives (on top of the fake retrieval)."""
    calls = []
    fake = answer.first_stage

    def spy(question, *args, **kwargs):
        calls.append(question)
        return fake(question, *args, **kwargs)

    monkeypatch.setattr(answer, "first_stage", spy)
    return calls


def test_startup_warms_up_retrieval_once_without_the_llm_or_the_log(
    fake_retrieval, retrieval_calls, tmp_path
):
    service, fake = make_service(fake_retrieval, tmp_path)
    with api(service) as client:
        assert retrieval_calls == [WARM_UP_QUERY]  # before any request
        assert client.get("/health").status_code == 200
    assert fake.requests == []  # no LLM call
    assert read_log(service.log.path) == []  # not a user request: not logged


def test_the_warm_up_runs_before_the_first_question(fake_retrieval, retrieval_calls, tmp_path):
    service, _ = make_service(fake_retrieval, tmp_path, ["A [1]."])
    with api(service) as client:
        client.post("/ask", json={"question": "Q?"})
    assert retrieval_calls == [WARM_UP_QUERY, "Q?"]
    assert [r["question"] for r in read_log(service.log.path)] == ["Q?"]


def test_warm_up_returns_its_duration_and_works_without_a_generator(
    fake_retrieval, retrieval_calls, tmp_path
):
    service, _ = make_service(fake_retrieval, tmp_path, client_error="GROQ_API_KEY is not set")
    assert service.warm_up() >= 0
    assert retrieval_calls == [WARM_UP_QUERY]


# --- query log ---------------------------------------------------------------------------------


def test_each_answer_is_logged_as_one_jsonl_line(fake_retrieval, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-test-key")
    service, _ = make_service(fake_retrieval, tmp_path, ["A [1].", ABSTAIN])
    with api(service) as client:
        first = client.post("/ask", json={"question": "Q1?"}).json()
        client.post("/ask", json={"question": "Q2?"})

    text = service.log.path.read_text(encoding="utf-8")
    assert "secret-test-key" not in text
    lines = text.splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["request_id"] == first["request_id"]
    assert record["question"] == "Q1?"
    assert record["setup"] == "hybrid"
    assert [r["chunk_id"] for r in record["retrieved"]] == [
        c["chunk_id"] for c in first["retrieved"]
    ]
    assert set(record["retrieved"][0]) == {"chunk_id", "rrf_score", "bm25_rank", "dense_rank"}
    assert record["cited"] == [1] and record["cited_chunk_ids"] == ["p:0020"]
    assert record["abstained"] is False and record["error"] is None
    assert set(record["latency_ms"]) == {"bm25", "dense", "fusion", "generation", "total"}
    assert record["model"] == "openai/gpt-oss-120b"
    assert record["request_hash"]
    assert json.loads(lines[1])["abstained"] is True


def test_the_log_directory_is_created(tmp_path):
    log = QueryLog(tmp_path / "new" / "dir" / "q.jsonl")
    log.append({"a": 1})
    assert read_log(log.path) == [{"a": 1}]


# --- frozen-config guard -----------------------------------------------------------------------


def test_the_committed_config_is_the_frozen_one():
    check_frozen_config(CONFIG)


@pytest.mark.parametrize(
    "path, value",
    [
        (("retrieval", "rrf_k"), 30),
        (("retrieval", "use_reranker"), True),
        (("retrieval", "use_reranker"), 0),  # 0 == False, but not the evaluated setting
        (("generation", "context_k"), 10),
        (("llm", "model"), "qwen/qwen3.8-27b"),
        (("llm", "temperature"), 0.7),
        (("chunking", "chunk_size"), 256),
        (("embedding", "model"), "BAAI/bge-base-en-v1.5"),
    ],
)
def test_any_other_config_is_refused(path, value):
    config = copy.deepcopy(CONFIG)
    config[path[0]][path[1]] = value
    with pytest.raises(FrozenConfigError, match=".".join(path)):
        check_frozen_config(config)


def test_a_missing_setting_is_refused():
    config = copy.deepcopy(CONFIG)
    del config["generation"]["context_k"]
    with pytest.raises(FrozenConfigError, match="generation.context_k is missing"):
        check_frozen_config(config)


def test_the_guard_runs_before_anything_is_loaded(monkeypatch):
    def must_not_load(config):
        raise AssertionError("indexes loaded for a refused config")

    monkeypatch.setattr(service_module, "load_indexes", must_not_load)
    config = copy.deepcopy(CONFIG)
    config["retrieval"]["use_reranker"] = True
    with pytest.raises(FrozenConfigError):
        RagService.from_config(config)


def test_a_missing_api_key_still_starts_the_service(monkeypatch, tmp_path):
    monkeypatch.setattr(service_module, "load_indexes", lambda config: {"by_id": {}})
    monkeypatch.setattr(service_module, "REPO_ROOT", tmp_path)

    def no_key(config, role):
        raise RuntimeError("GROQ_API_KEY is not set: add it to .env (see .env.example)")

    monkeypatch.setattr(service_module, "make_client", no_key)
    service = RagService.from_config(CONFIG)
    assert service.client is None
    assert service.health()["status"] == "degraded"

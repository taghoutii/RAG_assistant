import pytest

from ragpapers.config import load_config
from ragpapers.generation import answer
from ragpapers.generation.client import LLMClient
from ragpapers.generation.providers import FakeProvider

CONFIG = load_config()
ABSTAIN = CONFIG["abstention_text"]


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
def indexes(monkeypatch):
    by_id = {f"p:{n:04d}": chunk(n) for n in range(1, 21)}
    ranking = list(reversed(by_id))  # hybrid order: p:0020 first

    def fake_first_stage(question, indexes, fusion_depth, rrf_k):
        return {"bm25": [], "dense": [], "hybrid": ranking}, {"bm25": 1.0, "dense": 2.0}

    monkeypatch.setattr(answer, "first_stage", fake_first_stage)
    return {"by_id": by_id}


def test_answer_uses_the_top_context_k_chunks_and_cites_them(indexes):
    fake = FakeProvider(["Fact [2]."])
    result = answer.answer_question("Q?", indexes, LLMClient(fake, "m"), CONFIG)

    k = CONFIG["generation"]["context_k"]
    assert [c["chunk_id"] for c in result["chunks"]] == [
        f"p:{n:04d}" for n in range(20, 20 - k, -1)
    ]
    (request,) = fake.requests
    assert "[8] Paper > Section 13" in request.messages[1].content
    assert [s.chunk_id for s in result["cited"].sources] == ["p:0019"]
    assert set(result["latency_ms"]) == {"bm25", "dense", "generation_call"}


def test_abstention_comes_back_with_no_sources(indexes):
    client = LLMClient(FakeProvider([ABSTAIN]), "m")
    cited = answer.answer_question("Q?", indexes, client, CONFIG)["cited"]
    assert cited.abstained and cited.sources == []


def test_only_dev_questions_are_accepted():
    assert answer.dev_question("q101", CONFIG).startswith("How often does REALM")
    with pytest.raises(SystemExit, match="not a dev question"):
        answer.dev_question("q999", CONFIG)


def test_a_test_split_question_is_refused():
    from ragpapers.config import REPO_ROOT
    from ragpapers.evaluation.split import load_split

    test_id = sorted(load_split(REPO_ROOT / CONFIG["paths"]["eval_splits"], "test"))[0]
    with pytest.raises(SystemExit, match="held out"):
        answer.dev_question(test_id, CONFIG)

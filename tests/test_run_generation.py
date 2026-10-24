import json

import pytest

from ragpapers.config import load_config
from ragpapers.evaluation import run_generation
from ragpapers.evaluation.grading_sheet import grades_template, make_sheet
from ragpapers.generation.citations import cite_answer
from ragpapers.generation.client import LLMClient
from ragpapers.generation.providers import FakeProvider

CONFIG = load_config()
K = CONFIG["generation"]["context_k"]


@pytest.fixture
def indexes(monkeypatch):
    by_id = {
        f"p:{n}": {
            "chunk_id": f"p:{n}",
            "title": "T",
            "section": "S",
            "text": f"t{n}",
            "paper_id": "p",
            "year": 2020,
            "section_url": "u",
        }
        for n in range(30)
    }

    def fake_first_stage(question, indexes, fusion_depth, rrf_k):
        rankings = {
            "bm25": [f"p:{n}" for n in range(0, 30)],
            "dense": [f"p:{n}" for n in range(29, -1, -1)],
            "hybrid": [f"p:{n}" for n in range(10, 30)],
        }
        return rankings, {"bm25": 1.0, "dense": 2.0, "fusion": 0.1}

    def fake_rerank(question, ids, depth, by_id, reranker):
        return list(reversed(ids[:depth])), 50.0

    monkeypatch.setattr(run_generation, "first_stage", fake_first_stage)
    monkeypatch.setattr(run_generation, "rerank_top", fake_rerank)
    return {"by_id": by_id}


@pytest.mark.parametrize(
    ("setup", "first", "stages"),
    [
        ("bm25", "p:0", {"bm25"}),
        ("dense", "p:29", {"dense"}),
        ("hybrid", "p:10", {"bm25", "dense", "fusion"}),
        ("hybrid_rerank", "p:29", {"bm25", "dense", "fusion", "rerank"}),
    ],
)
def test_each_setup_returns_its_own_top_k(indexes, setup, first, stages):
    ids, ms = run_generation.retrieve_for_setup("Q?", setup, indexes, CONFIG)
    assert len(ids) == K and ids[0] == first and set(ms) == stages


def test_unknown_setup(indexes):
    with pytest.raises(ValueError):
        run_generation.retrieve_for_setup("Q?", "colbert", indexes, CONFIG)


def test_answer_record_keeps_what_the_judge_and_the_report_need():
    fake = FakeProvider(["Fact [2]. Other [9]."], model_reported="m-2026")
    response = LLMClient(fake, "m").generate([])
    chunks = [
        {
            "chunk_id": f"c{n}",
            "paper_id": "p",
            "title": "T",
            "year": 2020,
            "section": "S",
            "section_url": "u",
            "text": "x",
        }
        for n in range(8)
    ]
    cited = cite_answer(response.text, chunks, CONFIG["abstention_text"])
    record = run_generation.answer_record(
        {"id": "q1", "type": "factual"},
        "hybrid",
        [c["chunk_id"] for c in chunks],
        {"bm25": 1.0},
        response,
        cited,
    )
    assert record["chunk_ids"][1] == "c1" and record["cited"] == [2]
    assert record["invalid_citations"] == [9] and not record["abstained"]
    assert (record["requested_model"], record["model"]) == ("m", "m-2026")
    assert set(record["latency_ms"]) == {"bm25", "generation"}
    json.dumps(record)  # serialisable


def test_test_split_needs_the_final_flag():
    with pytest.raises(SystemExit, match="held out"):
        run_generation.main(["--setup", "hybrid", "--split", "test"])


def test_unknown_setup_is_refused_before_any_work():
    with pytest.raises(SystemExit, match="unknown setup"):
        run_generation.main(["--setup", "colbert"])


# --- grading sheet ------------------------------------------------------------------------------

QUESTIONS = {
    "q1": {"id": "q1", "type": "factual", "question": "What is X?", "reference_answer": "REF-ONE"},
    "q2": {"id": "q2", "type": "comparison", "question": "Compare?", "reference_answer": "REF-TWO"},
}
BY_ID = {"c1": {"chunk_id": "c1", "title": "T", "section": "S", "text": "passage text"}}
ANSWERS = [
    {"id": "q1", "type": "factual", "chunk_ids": ["c1"], "text": "X is Y [1].", "abstained": False},
    {
        "id": "q2",
        "type": "comparison",
        "chunk_ids": ["c1"],
        "text": CONFIG["abstention_text"],
        "abstained": True,
    },
]


def test_sheet_shows_references_only_after_part_1():
    sheet = make_sheet(ANSWERS, QUESTIONS, BY_ID)
    part2 = sheet.index("# Part 2")
    assert sheet.index("[1] T > S\npassage text") < part2
    assert sheet.index("REF-ONE") > part2
    assert "REF-TWO" not in sheet  # abstained: no correctness grade, reference never shown
    assert "Abstained: no grades needed" in sheet


def test_grades_template_has_null_labels():
    lines = [json.loads(line) for line in grades_template(ANSWERS).splitlines()]
    assert lines[0] == {
        "id": "q1",
        "type": "factual",
        "abstained": False,
        "faithfulness": None,
        "citation_correctness": None,
        "relevance": None,
        "correctness": None,
        "notes": "",
    }
    assert lines[1]["abstained"] is True


def test_generation_latency_from_an_old_cache_entry_is_not_measured():
    from dataclasses import replace

    response = LLMClient(FakeProvider(["Fact [1]."]), "m").generate([])
    cited = cite_answer(response.text, [], CONFIG["abstention_text"])
    q = {"id": "q1", "type": "factual"}
    fresh = run_generation.answer_record(q, "hybrid", [], {}, response, cited)
    assert fresh["latency_ms"]["generation"] == response.latency_ms and fresh["wait_ms"] == 0
    old = run_generation.answer_record(q, "hybrid", [], {}, replace(response, wait_ms=None), cited)
    assert old["latency_ms"]["generation"] is None and old["wait_ms"] is None

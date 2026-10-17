import pytest

from ragpapers.evaluation.run_retrieval import (
    aggregate,
    bootstrap_ci,
    make_table,
    paired_difference,
    run_setup,
    score_question,
)

CFG = {
    "ks": [1, 2],
    "top_n": 2,
    "quote_match_threshold": 90,
    "bootstrap_samples": 200,
    "split": "dev",
}
QUOTE = "the decoder attends over all encoded passages"


def chunk(cid, text):
    return {"chunk_id": cid, "text": text}


def test_score_question_uses_the_user_metrics():
    retrieved = [chunk("a", "nothing relevant here"), chunk("b", f"In FiD {QUOTE} jointly.")]
    result = score_question(retrieved, [[QUOTE]], ks=[1, 2], threshold=90)
    assert result["credited"] == [[], [0]]
    assert result["scores"]["rr"] == 0.5
    assert result["scores"]["recall@1"] == 0.0
    assert result["scores"]["recall@2"] == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [0.0, 1.0, 1.0, 0.5, 0.25, 1.0]
    low, high = bootstrap_ci(values, n_samples=500, seed=1)
    assert (low, high) == bootstrap_ci(values, n_samples=500, seed=1)
    assert low <= sum(values) / len(values) <= high
    assert bootstrap_ci([0.7, 0.7, 0.7], 100, 0) == pytest.approx((0.7, 0.7))


def test_aggregate_names_mrr():
    per_question = [
        {"scores": {"rr": 1.0, "recall@1": 1.0}},
        {"scores": {"rr": 0.0, "recall@1": 0.0}},
    ]
    out = aggregate(per_question, n_samples=100, seed=0)
    assert set(out) == {"mrr", "recall@1"}
    assert out["mrr"]["mean"] == 0.5


def test_run_setup_and_table():
    questions = [
        {"id": "q1", "type": "factual", "question": "?", "evidence": [{"quote": QUOTE}]},
        {
            "id": "q2",
            "type": "factual",
            "question": "?",
            "evidence": [{"quote": "absent quote text"}],
        },
    ]

    def retrieve(query, k):
        return [chunk("c1", QUOTE), chunk("c2", "other")][:k]

    setup = {"retriever": "bm25", "chunk_size": 150, "overlap": 30, "include_appendix": True}
    result = run_setup("bm25_test", setup, retrieve, questions, CFG, seed=0)
    assert result["n_questions"] == 2
    assert result["metrics"]["mrr"]["mean"] == 0.5
    assert result["per_question"][0]["retrieved"] == ["c1", "c2"]

    # make_table needs the k values it prints; reuse the result with renamed metrics.
    for k in (3, 5, 10):
        result["metrics"][f"recall@{k}"] = result["metrics"]["recall@2"]
    result["metrics"]["ndcg@10"] = result["metrics"]["ndcg@2"]
    table = make_table([result])
    assert "| bm25 | words | 150 / 30 | yes |" in table  # old results have no "unit"
    assert "2 answerable" in table


def fake_result(name, recalls, reference=False):
    per_question = [
        {"id": f"q{i}", "scores": {"recall@10": r, "rr": r}} for i, r in enumerate(recalls)
    ]
    return {"name": name, "per_question": per_question, "reference": reference}


def test_paired_difference_on_the_same_questions():
    a = fake_result("a", [1.0, 1.0, 0.0, 1.0])
    b = fake_result("b", [0.0, 1.0, 0.0, 0.0])
    diff, low, high = paired_difference(a, b, "recall@10", n_samples=500, seed=0)
    assert diff == 0.5
    assert 0.0 <= low <= diff <= high <= 1.0
    assert paired_difference(a, b, "mrr", 500, 0)[0] == 0.5


def test_paired_difference_refuses_different_question_sets():
    with pytest.raises(ValueError):
        paired_difference(fake_result("a", [1.0]), fake_result("b", [1.0, 0.0]), "mrr", 10, 0)

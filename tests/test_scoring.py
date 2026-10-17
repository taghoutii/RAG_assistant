import pytest

from ragpapers.evaluation.scoring import (
    aggregate,
    bootstrap_ci,
    fmt,
    paired_difference,
    score_question,
    write_result,
)

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


def fake_result(name, recalls):
    per_question = [
        {"id": f"q{i}", "scores": {"recall@10": r, "rr": r}} for i, r in enumerate(recalls)
    ]
    return {"name": name, "per_question": per_question}


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


def test_fmt_with_and_without_interval():
    metric = {"mean": 0.6111, "ci_low": 0.4611, "ci_high": 0.7722}
    assert fmt(metric) == "0.611"
    assert fmt(metric, with_ci=True) == "0.611 [0.46, 0.77]"


def test_write_result_path(tmp_path):
    path = write_result({"phase": "retrieval", "name": "hybrid", "x": 1}, tmp_path)
    assert path == tmp_path / "retrieval" / "hybrid.json"
    assert '"x": 1' in path.read_text(encoding="utf-8")

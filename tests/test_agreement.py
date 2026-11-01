import pytest

from ragpapers.evaluation.agreement import (
    cohen_kappa,
    judge_agreement,
    metric_agreement,
    raw_agreement,
)

RULE = {"metrics": ["faithfulness"], "kappa_min": 0.6, "skew_share": 0.85, "raw_min": 0.85}


def test_raw_agreement():
    assert raw_agreement([1, 0, 1, 1], [1, 1, 1, 1]) == 0.75
    with pytest.raises(ValueError):
        raw_agreement([1], [1, 0])


def test_kappa_textbook_example():
    # 50 items: a says yes 25, b says yes 30; they agree on 20 yes + 15 no.
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    # p_o = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = 0.4
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_perfect_chance_and_undefined():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None  # both raters use one label


def test_balanced_metric_uses_kappa():
    reference = [1, 0] * 10
    m = metric_agreement(reference, reference[:-2] + [0, 0], RULE)  # one disagreement
    assert not m["skewed"] and m["score"] == m["kappa"] and m["passed"]


def test_skewed_metric_uses_raw_agreement():
    reference = [1] * 18 + [0, 0]  # 90% ones > 85%
    judge = [1] * 20  # kappa 0, but raw 0.90
    m = metric_agreement(reference, judge, RULE)
    assert m["skewed"] and m["kappa"] == 0 and m["raw"] == 0.9 and m["passed"]
    assert not metric_agreement(reference, [0] * 4 + [1] * 16, RULE)["passed"]  # raw 0.70


def test_exactly_85_percent_is_not_skewed():
    reference = [1] * 17 + [0] * 3  # 85%: not "more than 85%"
    assert not metric_agreement(reference, reference, RULE)["skewed"]


def grades(labels, metric="faithfulness"):
    return {f"q{i}": {metric: v} for i, v in enumerate(labels)}


def test_missing_judge_labels_count_as_disagreement_and_abstentions_are_skipped():
    reference = grades([1, 0, 1, 0, None])  # q4: abstained, no grade
    judge = {
        "q0": {"faithfulness": 1},
        "q1": {"faithfulness": 0},
        "q2": {"faithfulness": None},
        "q3": {"faithfulness": 0},
    }
    result = judge_agreement(reference, judge, RULE)["metrics"]["faithfulness"]
    assert result["n"] == 4 and result["raw"] == 0.75


def test_poor_if_any_metric_fails_and_mean_score():
    rule = RULE | {"metrics": ["faithfulness", "relevance"]}
    reference = {f"q{i}": {"faithfulness": i % 2, "relevance": (i // 2) % 2} for i in range(20)}
    good = {i: dict(v) for i, v in reference.items()}
    bad_relevance = {
        i: {"faithfulness": v["faithfulness"], "relevance": 1} for i, v in reference.items()
    }

    poor = judge_agreement(reference, bad_relevance, rule)
    assert poor["poor"] and poor["metrics"]["faithfulness"]["passed"]
    assert not poor["metrics"]["relevance"]["passed"]
    good_result = judge_agreement(reference, good, rule)
    assert not good_result["poor"] and good_result["mean_score"] == 1
    assert poor["mean_score"] < good_result["mean_score"]

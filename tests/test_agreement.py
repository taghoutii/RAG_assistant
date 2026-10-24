import pytest

from ragpapers.evaluation.agreement import (
    choose_judge,
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
    human = [1, 0] * 10
    m = metric_agreement(human, human[:-2] + [0, 0], RULE)  # one disagreement
    assert not m["skewed"] and m["score"] == m["kappa"] and m["passed"]


def test_skewed_metric_uses_raw_agreement():
    human = [1] * 18 + [0, 0]  # 90% ones > 85%
    judge = [1] * 20  # kappa 0, but raw 0.90
    m = metric_agreement(human, judge, RULE)
    assert m["skewed"] and m["kappa"] == 0 and m["raw"] == 0.9 and m["passed"]
    assert not metric_agreement(human, [0] * 4 + [1] * 16, RULE)["passed"]  # raw 0.70


def test_exactly_85_percent_is_not_skewed():
    human = [1] * 17 + [0] * 3  # 85%: not "more than 85%"
    assert not metric_agreement(human, human, RULE)["skewed"]


def grades(labels, metric="faithfulness"):
    return {f"q{i}": {metric: v} for i, v in enumerate(labels)}


def test_missing_judge_labels_count_as_disagreement_and_abstentions_are_skipped():
    human = grades([1, 0, 1, 0, None])  # q4: abstained, no grade
    judge = {
        "q0": {"faithfulness": 1},
        "q1": {"faithfulness": 0},
        "q2": {"faithfulness": None},
        "q3": {"faithfulness": 0},
    }
    result = judge_agreement(human, judge, RULE)["metrics"]["faithfulness"]
    assert result["n"] == 4 and result["raw"] == 0.75


def test_poor_if_any_metric_fails_and_choice_by_mean_score():
    rule = RULE | {"metrics": ["faithfulness", "relevance"]}
    human = {f"q{i}": {"faithfulness": i % 2, "relevance": (i // 2) % 2} for i in range(20)}
    good = {i: dict(v) for i, v in human.items()}
    bad_relevance = {
        i: {"faithfulness": v["faithfulness"], "relevance": 1} for i, v in human.items()
    }

    primary = judge_agreement(human, bad_relevance, rule)
    assert primary["poor"] and primary["metrics"]["faithfulness"]["passed"]
    fallback = judge_agreement(human, good, rule)
    assert not fallback["poor"] and fallback["mean_score"] == 1
    assert choose_judge(primary, fallback) == "judge_fallback"
    assert choose_judge(fallback, None) == "judge"  # not poor: no fallback needed
    assert choose_judge(primary, primary) == "judge"  # tie: the primary stays
    with pytest.raises(ValueError):
        choose_judge(primary, None)

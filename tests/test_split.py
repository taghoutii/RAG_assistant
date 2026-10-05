from collections import Counter

from ragpapers.evaluation.split import stratified_split

QUESTIONS = (
    [{"id": f"q{i:03d}", "type": "factual"} for i in range(1, 41)]
    + [{"id": f"q{i:03d}", "type": "comparison"} for i in range(41, 61)]
    + [{"id": f"q{i:03d}", "type": "unanswerable"} for i in range(61, 76)]
)
TYPES = {q["id"]: q["type"] for q in QUESTIONS}


def test_split_is_disjoint_and_complete():
    split = stratified_split(QUESTIONS, test_fraction=0.4, seed=42)
    assert not set(split["dev"]) & set(split["test"])
    assert set(split["dev"]) | set(split["test"]) == set(TYPES)


def test_split_keeps_the_type_mix():
    split = stratified_split(QUESTIONS, test_fraction=0.4, seed=42)
    test_counts = Counter(TYPES[i] for i in split["test"])
    assert test_counts == {"factual": 16, "comparison": 8, "unanswerable": 6}


def test_split_is_deterministic_for_a_seed():
    a = stratified_split(QUESTIONS, 0.4, seed=42)
    b = stratified_split(list(reversed(QUESTIONS)), 0.4, seed=42)
    c = stratified_split(QUESTIONS, 0.4, seed=7)
    assert a == b
    assert a != c

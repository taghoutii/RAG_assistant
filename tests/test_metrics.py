import math

import pytest

from ragpapers.evaluation.metrics import credit_quotes, ndcg_at_k, recall_at_k, reciprocal_rank

THRESHOLD = 90
K = "model size should grow faster than data"  # quote 0
C1 = "scaled in equal proportions with training tokens"  # quote 1
C2 = "a 70B model trained on 1.4 trillion tokens"  # quote 2
QUOTES = [K, C1, C2]

# The worked example from the Phase 3 explanation: rank 3 repeats C1 (overlap duplicate).
CHUNKS = [
    "Unrelated text about tokenizers.",
    f"Our finding: {C1}, not otherwise.",
    f"... {C1}, not otherwise. More text follows here.",
    f"Kaplan et al. argue that {K}.",
    f"We repeat that it is {C1}; to test it we built {C2}.",
]
CREDITED = [[], [1], [], [0], [2]]


# --- credit_quotes -----------------------------------------------------------------------------


def test_credit_quotes_worked_example():
    assert credit_quotes(CHUNKS, QUOTES, THRESHOLD) == CREDITED


def test_credit_quotes_one_chunk_can_credit_several_quotes():
    chunks = [f"{K} and also {C2}", f"{C1}"]
    assert credit_quotes(chunks, QUOTES, THRESHOLD) == [[0, 2], [1]]


def test_credit_quotes_ignores_case_and_ref_markers():
    chunks = ["Model size should grow faster [ref] than data."]
    assert credit_quotes(chunks, [K], THRESHOLD) == [[0]]


def test_credit_quotes_no_chunks():
    assert credit_quotes([], QUOTES, THRESHOLD) == []


# --- recall_at_k -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"), [(1, 0.0), (2, 1 / 3), (3, 1 / 3), (4, 2 / 3), (5, 1.0)]
)
def test_recall_at_k(k, expected):
    assert recall_at_k(CREDITED, n_quotes=3, k=k) == pytest.approx(expected)


def test_recall_with_fewer_chunks_than_k():
    assert recall_at_k([[0]], n_quotes=2, k=10) == pytest.approx(0.5)


@pytest.mark.parametrize(("n_quotes", "k"), [(0, 5), (3, 0)])
def test_recall_rejects_invalid_input(n_quotes, k):
    with pytest.raises(ValueError):
        recall_at_k(CREDITED, n_quotes=n_quotes, k=k)


# --- reciprocal_rank ---------------------------------------------------------------------------


def test_reciprocal_rank_worked_example():
    assert reciprocal_rank(CREDITED) == pytest.approx(0.5)


def test_reciprocal_rank_first_and_none():
    assert reciprocal_rank([[0], []]) == 1.0
    assert reciprocal_rank([[], [], []]) == 0.0
    assert reciprocal_rank([]) == 0.0


# --- ndcg_at_k ---------------------------------------------------------------------------------


def test_ndcg_worked_example():
    dcg = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(CREDITED, n_quotes=3, k=5) == pytest.approx(dcg / idcg)  # about 0.68


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([[0], [1], [], []], n_quotes=2, k=4) == pytest.approx(1.0)


def test_ndcg_ideal_is_capped_at_k():
    # 3 quotes but k = 2: the ideal has only 2 relevant slots, so 2 hits at the top score 1.
    assert ndcg_at_k([[0], [1], [2]], n_quotes=3, k=2) == pytest.approx(1.0)


def test_ndcg_nothing_found_is_zero():
    assert ndcg_at_k([[], []], n_quotes=1, k=2) == 0.0


@pytest.mark.parametrize(("n_quotes", "k"), [(0, 5), (3, 0)])
def test_ndcg_rejects_invalid_input(n_quotes, k):
    with pytest.raises(ValueError):
        ndcg_at_k(CREDITED, n_quotes=n_quotes, k=k)

import pytest

from ragpapers.retrieval.hybrid import reciprocal_rank_fusion

# The worked example from the Phase 4 explanation: BM25 and dense top-3 for one query.
BM25 = ["c1", "c2", "c3"]
DENSE = ["c3", "c1", "c4"]


def ids(fused):
    return [chunk_id for chunk_id, _ in fused]


def scores(fused):
    return dict(fused)


# --- the worked example ------------------------------------------------------------------------


def test_worked_example_order():
    # c3 (dense #1, BM25 #3) beats c2 (BM25 #2 only): agreement between retrievers wins.
    assert ids(reciprocal_rank_fusion([BM25, DENSE])) == ["c1", "c3", "c2", "c4"]


def test_worked_example_scores():
    fused = scores(reciprocal_rank_fusion([BM25, DENSE], rrf_k=60))
    assert fused["c1"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c3"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["c2"] == pytest.approx(1 / 62)
    assert fused["c4"] == pytest.approx(1 / 63)


def test_default_rrf_k_is_60():
    assert reciprocal_rank_fusion([BM25, DENSE]) == reciprocal_rank_fusion([BM25, DENSE], rrf_k=60)


# --- formula details ---------------------------------------------------------------------------


def test_ranks_start_at_one():
    # Off-by-one check: the top chunk of a single list scores 1 / (rrf_k + 1), not 1 / rrf_k.
    assert scores(reciprocal_rank_fusion([["a"]], rrf_k=10))["a"] == pytest.approx(1 / 11)


def test_rrf_k_zero_gives_plain_reciprocal_rank():
    fused = scores(reciprocal_rank_fusion([["a", "b", "c"]], rrf_k=0))
    assert fused == pytest.approx({"a": 1.0, "b": 1 / 2, "c": 1 / 3})


def test_rrf_k_changes_the_winner():
    # "solo" is #1 in one list only; "both" is #3 in two lists.
    rankings = [["solo", "x", "both"], ["y", "z", "both"]]
    # Small k: a single first place outweighs two third places (1/1 > 2/3).
    assert ids(reciprocal_rank_fusion(rankings, rrf_k=0))[0] == "solo"
    # Large k: agreement wins (2/63 > 1/61).
    assert ids(reciprocal_rank_fusion(rankings, rrf_k=60))[0] == "both"


def test_more_than_two_rankings():
    fused = scores(reciprocal_rank_fusion([["a", "b"], ["b"], ["b", "a"]], rrf_k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


# --- output shape ------------------------------------------------------------------------------


def test_returns_the_union_of_all_ids_once_each():
    fused = ids(reciprocal_rank_fusion([["a", "b", "c"], ["d", "a"]]))
    assert sorted(fused) == ["a", "b", "c", "d"]


def test_scores_are_sorted_descending():
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"], ["d", "e", "b"], ["f"]])
    values = [score for _, score in fused]
    assert values == sorted(values, reverse=True)


def test_single_ranking_keeps_its_order():
    assert ids(reciprocal_rank_fusion([["z", "a", "m"]])) == ["z", "a", "m"]


def test_lists_of_different_lengths():
    fused = ids(reciprocal_rank_fusion([["a"], ["b", "c", "d", "e"]]))
    assert fused[:2] == ["a", "b"]  # tie at 1/61, broken by chunk ID
    assert fused[2:] == ["c", "d", "e"]


def test_ties_are_broken_by_chunk_id():
    # Mirror-image lists give "y" and "x" exactly the same score.
    assert ids(reciprocal_rank_fusion([["y", "x"], ["x", "y"]])) == ["x", "y"]


# --- edge cases --------------------------------------------------------------------------------


def test_no_rankings():
    assert reciprocal_rank_fusion([]) == []


def test_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_ranking_is_ignored():
    assert reciprocal_rank_fusion([BM25, []]) == reciprocal_rank_fusion([BM25])


def test_does_not_modify_its_input():
    rankings = [list(BM25), list(DENSE)]
    reciprocal_rank_fusion(rankings)
    assert rankings == [BM25, DENSE]


def test_rejects_negative_rrf_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([BM25], rrf_k=-1)


def test_rejects_duplicate_id_within_one_list():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a", "b", "a"], ["c"]])


def test_duplicate_error_names_the_id_and_the_ranking():
    with pytest.raises(ValueError, match=r"'b' appears twice in ranking 1"):
        reciprocal_rank_fusion([["a"], ["c", "b", "d", "b"]])


def test_same_id_in_different_lists_is_fine():
    # Appearing in several lists is the whole point; only duplicates WITHIN a list are errors.
    assert ids(reciprocal_rank_fusion([["a"], ["a"]])) == ["a"]

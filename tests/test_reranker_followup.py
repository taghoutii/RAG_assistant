from ragpapers.evaluation.reranker_followup import (
    adopt,
    biggest_losses,
    choose_variant,
    fuse_with_hybrid,
)


def test_fuse_with_hybrid_keeps_chunks_outside_the_pool():
    hybrid = ["h1", "h2", "h3"]
    reranked = ["h2", "h1"]  # the reranker only saw the top 2
    fused = fuse_with_hybrid(hybrid, reranked, rrf_k=60)
    assert sorted(fused) == ["h1", "h2", "h3"]
    assert fused[-1] == "h3"  # only its hybrid contribution


def test_fuse_with_hybrid_agreement_wins():
    # "x" is #2 in hybrid and #1 for the reranker; "y" is #1 in hybrid and #3 for the reranker.
    fused = fuse_with_hybrid(["y", "x", "z"], ["x", "z", "y"], rrf_k=60)
    assert fused[0] == "x"


def test_choose_variant_best_value():
    assert choose_variant({"a": 0.5, "b": 0.6, "c": 0.55}, ["a", "b", "c"]) == "b"


def test_choose_variant_ties_go_to_the_earliest():
    assert choose_variant({"a": 0.6, "b": 0.6, "c": 0.6}, ["a", "b", "c"]) == "a"
    assert choose_variant({"a": 0.5, "b": 0.6, "c": 0.6}, ["a", "b", "c"]) == "b"


def test_adopt_needs_the_full_gain():
    assert adopt(0.608, 0.578, 0.03)  # exactly +0.03 counts (float slack)
    assert not adopt(0.600, 0.578, 0.03)
    assert not adopt(0.544, 0.578, 0.03)


def test_biggest_losses_order_and_ties():
    hybrid = {"q1": 1.0, "q2": 1.0, "q3": 0.5, "q4": 0.2}
    variant = {"q1": 0.5, "q2": 0.5, "q3": 0.1, "q4": 1.0}
    # losses: q1 0.5, q2 0.5, q3 0.4; q4 gained and is never listed
    assert biggest_losses(hybrid, variant, n=2) == ["q1", "q2"]
    assert biggest_losses(hybrid, variant, n=10) == ["q1", "q2", "q3"]

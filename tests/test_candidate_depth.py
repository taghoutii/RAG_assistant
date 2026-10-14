import pytest

from ragpapers.evaluation.candidate_depth import choose_depth, hybrid_pool, p95, pool_recall
from ragpapers.retrieval.rerank import Reranker

QUOTE = "the decoder attends over all encoded passages"
DEPTHS = [20, 50, 100]
FAST = {20: 300.0, 50: 700.0, 100: 1500.0}


# --- hybrid_pool -------------------------------------------------------------------------------


def test_hybrid_pool_has_exactly_n_ids():
    bm25 = [f"b{i}" for i in range(10)]
    dense = [f"d{i}" for i in range(10)]
    assert len(hybrid_pool(bm25, dense, n=4, rrf_k=60)) == 4


def test_hybrid_pool_only_fuses_each_retrievers_top_n():
    # "deep" is #1 in dense but beyond BM25's top 2, so only dense counts it; "shared" is in both.
    bm25 = ["shared", "b1", "deep"]
    dense = ["deep", "shared", "d2"]
    assert hybrid_pool(bm25, dense, n=2, rrf_k=60) == ["shared", "deep"]


def test_hybrid_pool_agreement_ranks_first():
    assert hybrid_pool(["a", "x"], ["y", "a"], n=3, rrf_k=60)[0] == "a"


# --- pool_recall -------------------------------------------------------------------------------


def test_pool_recall_counts_quotes_anywhere_in_the_pool():
    texts = ["unrelated", "also unrelated", f"In FiD {QUOTE} jointly."]
    assert pool_recall(texts, [QUOTE, "a quote that is nowhere at all"], 90) == 0.5


def test_pool_recall_empty_pool():
    assert pool_recall([], [QUOTE], 90) == 0.0


# --- choose_depth ------------------------------------------------------------------------------


def test_choose_depth_stops_where_recall_flattens():
    recall = {20: 0.60, 50: 0.72, 100: 0.74}  # +0.12 then +0.02
    assert choose_depth(DEPTHS, recall, FAST, 0.05, 2000)[0] == 50


def test_choose_depth_flat_from_the_start():
    recall = {20: 0.70, 50: 0.72, 100: 0.80}  # the first step already gains < 0.05
    assert choose_depth(DEPTHS, recall, FAST, 0.05, 2000)[0] == 20


def test_choose_depth_never_flattens_takes_the_largest():
    recall = {20: 0.50, 50: 0.60, 100: 0.70}
    assert choose_depth(DEPTHS, recall, FAST, 0.05, 2000)[0] == 100


def test_choose_depth_latency_caps_the_choice():
    recall = {20: 0.50, 50: 0.60, 100: 0.70}
    depth, reason = choose_depth(DEPTHS, recall, FAST, 0.05, latency_budget_ms=1000)
    assert depth == 50
    assert "exceeds" in reason


def test_choose_depth_nothing_within_budget():
    recall = {20: 0.50, 50: 0.60, 100: 0.70}
    assert choose_depth(DEPTHS, recall, FAST, 0.05, latency_budget_ms=100)[0] == 20


def test_choose_depth_gain_exactly_at_threshold_is_not_flat():
    recall = {20: 0.50, 50: 0.55, 100: 0.56}  # +0.05 is not "< 0.05"
    assert choose_depth(DEPTHS, recall, FAST, 0.05, 2000)[0] == 50


def test_p95():
    assert p95(list(range(1, 101))) == pytest.approx(95.05)


# --- Reranker (fake model: CI downloads nothing) -----------------------------------------------


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the text."""

    def predict(self, pairs, batch_size, show_progress_bar):
        return [sum(w in text for w in query.split()) for query, text in pairs]


def make_reranker():
    reranker = Reranker("fake")
    reranker._model = FakeCrossEncoder()
    return reranker


def chunk(cid, text):
    return {"chunk_id": cid, "dense_header": "Paper > Section", "text": text}


def test_rerank_orders_by_score():
    chunks = [chunk("a", "nothing"), chunk("b", "lora rank"), chunk("c", "lora")]
    ranked = make_reranker().rerank("lora rank", chunks)
    assert [c["chunk_id"] for c, _ in ranked] == ["b", "c", "a"]
    assert [s for _, s in ranked] == [2.0, 1.0, 0.0]


def test_rerank_ties_keep_input_order():
    chunks = [chunk("x", "same"), chunk("y", "same")]
    assert [c["chunk_id"] for c, _ in make_reranker().rerank("q", chunks)] == ["x", "y"]


def test_rerank_sees_the_header():
    # The header "Paper > Section" contains "Section", so the score is 1, not 0.
    assert make_reranker().rerank("Section", [chunk("a", "body")])[0][1] == 1.0


def test_rerank_empty_pool():
    assert make_reranker().rerank("q", []) == []

from ragpapers.evaluation.candidate_depth import choose_rerank_depth, pool_recall
from ragpapers.retrieval.pipeline import rerank_top
from ragpapers.retrieval.rerank import Reranker

QUOTE = "the decoder attends over all encoded passages"


# --- pool_recall -------------------------------------------------------------------------------


def test_pool_recall_counts_quotes_anywhere_in_the_pool():
    texts = ["unrelated", "also unrelated", f"In FiD {QUOTE} jointly."]
    assert pool_recall(texts, [[QUOTE], ["a quote that is nowhere at all"]], 90) == 0.5


def test_pool_recall_empty_pool():
    assert pool_recall([], [[QUOTE]], 90) == 0.0


# --- choose_rerank_depth -----------------------------------------------------------------------


def test_choose_rerank_depth_takes_the_best_when_others_are_far():
    assert choose_rerank_depth({20: 0.60, 50: 0.70, 100: 0.75}, tolerance=0.02) == 100


def test_choose_rerank_depth_smallest_within_tolerance():
    assert choose_rerank_depth({20: 0.70, 50: 0.75, 100: 0.76}, tolerance=0.02) == 50


def test_choose_rerank_depth_exactly_at_tolerance_counts_as_within():
    # 0.80 - 0.78 is 0.020000000000000018 in floating point; it must still count as "within".
    assert choose_rerank_depth({20: 0.78, 50: 0.80, 100: 0.80}, tolerance=0.02) == 20


def test_choose_rerank_depth_ties_go_to_the_smaller_depth():
    assert choose_rerank_depth({20: 0.70, 50: 0.80, 100: 0.80}, tolerance=0.0) == 50


def test_choose_rerank_depth_best_can_be_the_smallest():
    # Deeper pools can add distractors that the reranker promotes.
    assert choose_rerank_depth({20: 0.80, 50: 0.75, 100: 0.70}, tolerance=0.02) == 20


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


def test_rerank_top_reranks_only_the_top_m():
    by_id = {
        "a": chunk("a", "nothing"),
        "b": chunk("b", "lora"),
        "c": chunk("c", "lora rank"),  # best match, but beyond M=2
    }
    ids, ms = rerank_top("lora rank", ["a", "b", "c"], 2, by_id, make_reranker())
    assert ids == ["b", "a"]
    assert ms >= 0

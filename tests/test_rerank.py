from ragpapers.retrieval.pipeline import rerank_top
from ragpapers.retrieval.rerank import Reranker


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the text (CI downloads no model)."""

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

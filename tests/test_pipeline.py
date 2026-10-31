import pytest

from ragpapers.retrieval.hybrid import reciprocal_rank_fusion
from ragpapers.retrieval.pipeline import first_stage


class FakeSearch:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, depth):
        return [({"chunk_id": i}, 1.0) for i in self.ids[:depth]]


class FakeEmbedder:
    def encode_query(self, query):
        return [0.0]


def test_first_stage_keeps_the_rrf_scores_without_changing_the_ranking():
    bm25, dense = ["a", "b", "c"], ["c", "d", "a"]
    indexes = {"bm25": FakeSearch(bm25), "dense": FakeSearch(dense), "embedder": FakeEmbedder()}
    rankings, ms = first_stage("q", indexes, fusion_depth=3, rrf_k=60)

    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    assert rankings["hybrid"] == [i for i, _ in fused]
    assert rankings["hybrid_scores"] == dict(fused)
    assert rankings["hybrid_scores"]["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert set(ms) == {"bm25", "dense", "fusion"}

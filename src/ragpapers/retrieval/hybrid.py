"""Hybrid retrieval: Reciprocal Rank Fusion (RRF) of several rankings, e.g. BM25 + dense.

RRF ignores the retrievers' raw scores (BM25 scores and cosine similarities are on unrelated
scales) and uses only each chunk's RANK in each list:

    score(chunk) = sum over the lists that contain it of  1 / (rrf_k + rank)

with rank starting at 1. A chunk missing from a list gets nothing from that list.
Reference: Cormack, Clarke and Buettcher (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods", which uses rrf_k = 60.
"""


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be >= 0, got {rrf_k}")

    scores: dict[str, float] = {}  # chunk ID -> running fused score
    for list_index, ranking in enumerate(rankings):
        if len(set(ranking)) != len(ranking):
            duplicate = next(c for i, c in enumerate(ranking) if c in ranking[:i])
            raise ValueError(f"chunk ID {duplicate!r} appears twice in ranking {list_index}")
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    # Highest score first; equal scores broken by chunk ID, ascending
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

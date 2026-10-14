"""Hybrid retrieval: Reciprocal Rank Fusion (RRF) of several rankings, e.g. BM25 + dense.

RRF ignores the retrievers' raw scores (BM25 scores and cosine similarities are on unrelated
scales) and uses only each chunk's RANK in each list:

    score(chunk) = sum over the lists that contain it of  1 / (rrf_k + rank)

with rank starting at 1. A chunk missing from a list gets nothing from that list.
Reference: Cormack, Clarke and Buettcher (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods", which uses rrf_k = 60.
"""


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Fuse several rankings of chunk IDs into one, best first.

    Args:
        rankings: one list per retriever, each a list of chunk IDs ordered best first
            (position 0 is rank 1). Lists may have different lengths, and may be empty.
        rrf_k: the smoothing constant. Larger values shrink the gap between rank 1 and rank 2,
            so agreement between retrievers matters more than a single top position.
            0 is allowed (score = 1 / rank).

    Returns:
        (chunk_id, fused_score) pairs for EVERY chunk ID that appears in at least one list,
        sorted by fused score, highest first. Ties are broken by chunk ID, ascending, so the
        output is deterministic.

    Raises:
        ValueError: if rrf_k is negative, or if a chunk ID appears twice in the same list.

    The input lists are not modified.
    """
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

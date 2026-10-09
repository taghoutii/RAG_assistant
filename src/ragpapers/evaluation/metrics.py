"""Retrieval metrics: Recall@k, reciprocal rank (MRR) and nDCG@k.

Relevance is judged against the gold evidence quotes of a question (see eval/README.md):
- a retrieved chunk is relevant if it contains an evidence quote (fuzzy match, `contains_quote`);
- each quote is credited only ONCE, at the highest-ranked chunk that contains it. A later chunk
  containing an already-credited quote (e.g. an overlapping neighbour) earns nothing.

The pipeline for one question is:
    credited = credit_quotes(retrieved_chunk_texts, quotes, threshold)
    recall_at_k(credited, len(quotes), k)
    reciprocal_rank(credited)
    ndcg_at_k(credited, len(quotes), k)
MRR over a question set is the mean of `reciprocal_rank` over its (answerable) questions.
Unanswerable questions have no quotes and are excluded from these metrics.
"""

import math

from ragpapers.evaluation.matching import contains_quote


def credit_quotes(chunk_texts: list[str], quotes: list[str], threshold: float) -> list[list[int]]:
    """For each retrieved chunk, in rank order, the quote indices it is the FIRST to contain.

    Args:
        chunk_texts: retrieved chunk texts, best-ranked first.
        quotes: the question's evidence quotes.
        threshold: score (0-100) passed to `contains_quote` to decide a chunk contains a quote.

    Returns:
        A list with one entry per chunk (same length as `chunk_texts`). Each entry is the sorted
        list of quote indices first found in that chunk; [] if it adds no new quote.

    Example: quotes [A, B]; chunks [contains A, contains A, contains A and B] -> [[0], [], [1]]
    """
    credited: set[int] = set()  # quote indices already claimed by a higher-ranked chunk
    result: list[list[int]] = []

    for text in chunk_texts:  # rank order: best chunk first
        new: list[int] = []
        for i, quote in enumerate(quotes):
            if i in credited:
                continue  # already claimed: an overlapping neighbour earns nothing
            if contains_quote(quote, text, threshold):
                new.append(i)
                credited.add(i)
        result.append(new)  # one entry per chunk, even if empty

    return result


def recall_at_k(credited: list[list[int]], n_quotes: int, k: int) -> float:
    """Fraction of the question's quotes credited within the top k chunks.

    Raises ValueError if n_quotes == 0 (unanswerable questions have no recall) or k < 1.
    If fewer than k chunks were retrieved, use the ones there are.
    """
    if n_quotes == 0:
        raise ValueError("recall is undefined for a question without quotes")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    found = sum(len(entry) for entry in credited[:k])
    return found / n_quotes


def reciprocal_rank(credited: list[list[int]]) -> float:
    """1 / rank of the first chunk that credits at least one quote (ranks start at 1).

    Returns 0.0 if no chunk credits a quote.
    """
    for rank, entry in enumerate(credited, start=1):
        if entry:  # a non-empty list: this chunk credits at least one quote
            return 1.0 / rank
    return 0.0


def ndcg_at_k(credited: list[list[int]], n_quotes: int, k: int) -> float:
    """Normalized discounted cumulative gain over the top k chunks, with binary gains.

    gain(rank) = 1 if the chunk at that rank credits at least one quote, else 0.
    DCG@k  = sum over ranks r = 1..k of gain(r) / log2(r + 1)
    IDCG@k = the DCG of an ideal ranking: min(k, n_quotes) gains of 1 at the top ranks.
    nDCG@k = DCG@k / IDCG@k, in [0, 1].

    Note: one chunk can contain two quotes but still has gain 1, so a ranking can find every quote
    and still score below 1. That is accepted: the ideal assumes one chunk per quote.
    Raises ValueError if n_quotes == 0 or k < 1.
    """
    if n_quotes == 0:
        raise ValueError("nDCG is undefined for a question without quotes")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    dcg = sum(1 / math.log2(rank + 1) for rank, entry in enumerate(credited[:k], start=1) if entry)
    ideal_hits = min(k, n_quotes)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg

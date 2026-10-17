"""Retrieval metrics: Recall@k, reciprocal rank (MRR) and nDCG@k.

Relevance is judged against the gold evidence of a question (see eval/README.md). Each evidence
item is a GROUP of quotes: its original quote plus any approved alternatives (the same fact,
stated elsewhere in the same paper). See `evidence_groups` in evaluation/questions.py.
- a retrieved chunk is relevant if it contains any quote of a group (fuzzy match,
  `contains_quote`);
- each group is credited only ONCE, at the highest-ranked chunk that contains any of its quotes.
  A later chunk containing an already-credited group (e.g. an overlapping neighbour, or the
  other place the paper states the fact) earns nothing.

The pipeline for one question is:
    groups = evidence_groups(question)
    credited = credit_quotes(retrieved_chunk_texts, groups, threshold)
    recall_at_k(credited, len(groups), k)
    reciprocal_rank(credited)
    ndcg_at_k(credited, len(groups), k)
MRR over a question set is the mean of `reciprocal_rank` over its (answerable) questions.
Unanswerable questions have no evidence and are excluded from these metrics.
A group of one quote behaves exactly like the original single-quote scoring.
"""

import math

from ragpapers.evaluation.matching import contains_quote


def credit_quotes(
    chunk_texts: list[str], quote_groups: list[list[str]], threshold: float
) -> list[list[int]]:
    """For each retrieved chunk, in rank order, the evidence groups it is the FIRST to contain.

    Args:
        chunk_texts: retrieved chunk texts, best-ranked first.
        quote_groups: one list of quotes per evidence item (original first, then alternatives).
            A chunk contains a group if it contains ANY of the group's quotes.
        threshold: score (0-100) passed to `contains_quote` to decide a chunk contains a quote.

    Returns:
        A list with one entry per chunk (same length as `chunk_texts`). Each entry is the sorted
        list of group indices first found in that chunk; [] if it adds no new group.

    Example: groups [[A], [B, B2]]; chunks [contains A, contains B2, contains B]
        -> [[0], [1], []]   (group 1 is credited at B2; the later chunk with B earns nothing)
    """
    if any(isinstance(group, str) for group in quote_groups):
        # A bare string would be iterated character by character and match almost anything.
        raise TypeError("quote_groups must be lists of quotes, e.g. [[quote]], not [quote]")

    credited: set[int] = set()  # group indices already claimed by a higher-ranked chunk
    result: list[list[int]] = []

    for text in chunk_texts:  # rank order: best chunk first
        new: list[int] = []
        for i, group in enumerate(quote_groups):
            if i in credited:
                continue  # already claimed: an overlapping neighbour earns nothing
            if any(contains_quote(quote, text, threshold) for quote in group):
                new.append(i)
                credited.add(i)
        result.append(new)  # one entry per chunk, even if empty

    return result


def recall_at_k(credited: list[list[int]], n_quotes: int, k: int) -> float:
    if n_quotes == 0:
        raise ValueError("recall is undefined for a question without quotes")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    found = sum(len(entry) for entry in credited[:k])
    return found / n_quotes


def reciprocal_rank(credited: list[list[int]]) -> float:
    for rank, entry in enumerate(credited, start=1):
        if entry:  # a non-empty list: this chunk credits at least one quote
            return 1.0 / rank
    return 0.0


def ndcg_at_k(credited: list[list[int]], n_quotes: int, k: int) -> float:
    if n_quotes == 0:
        raise ValueError("nDCG is undefined for a question without quotes")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    dcg = sum(1 / math.log2(rank + 1) for rank, entry in enumerate(credited[:k], start=1) if entry)
    ideal_hits = min(k, n_quotes)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg

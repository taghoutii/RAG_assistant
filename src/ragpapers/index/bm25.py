"""Keyword index: BM25 (rank_bm25) over each chunk's `bm25_text` (full header + chunk text)."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ragpapers.chunking.build import bm25_text

TOKEN = re.compile(r"[a-z0-9]+")
# Common English function words; they carry no topic signal for BM25.
_STOPWORD_TEXT = """
    a an and are as at be by can did do does for from has have how i in is it its of on or our
    than that the their then there these they this those to was we were what when where which
    who why will with you your
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stopwords.

    Example: "How does Self-Attention work?" -> ["self", "attention", "work"]
    """
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """BM25 over a fixed list of chunks. Search returns (chunk, score), best first."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(bm25_text(c)) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

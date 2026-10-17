"""Cross-encoder reranking: rescore a candidate pool by reading query and chunk together.

The first-stage retrievers score the query and each chunk separately (BM25 term statistics, or
two independent embeddings). A cross-encoder feeds "query [SEP] chunk" through one transformer,
so every query word can attend to every chunk word. That is far more accurate, but it costs one
forward pass per candidate, which is why it only reranks a small pool.

The reranker sees the same text as the embedding model (`dense_text`: short header + chunk).
Pairs longer than the model's window (512 tokens) are truncated from the end of the chunk.
"""

from ragpapers.chunking.build import dense_text


class Reranker:
    """Thin wrapper around a sentence-transformers CrossEncoder (loaded lazily on first use)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # slow import, only when needed

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text (higher = more relevant), in input order."""
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
        """`chunks` reordered by cross-encoder score, best first (ties keep the input order)."""
        scores = self.score(query, [dense_text(c) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])  # sorted() is stable
        return [(chunks[i], scores[i]) for i in order]

"""Dense index: bge-small embeddings stored in Qdrant (local embedded mode, no server)."""

import hashlib
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from ragpapers.chunking.build import dense_text


class Embedder:
    """Thin wrapper around a sentence-transformers model (loaded lazily on first use)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import, only when needed

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=True
        )

    def encode_query(self, query: str, instruction: str = "") -> np.ndarray:
        return self.model.encode(instruction + query, normalize_embeddings=True)


def load_or_compute_embeddings(
    chunks: list[dict], embed_passages, model_name: str, cache_path: Path
) -> np.ndarray:
    """Vectors for `chunks` in the same order, from the cache when it matches.

    Args:
        embed_passages: function list[str] -> np.ndarray (e.g. Embedder.encode_passages).
        cache_path: <name>.npy; its metadata (model, chunk ids, texts hash) is in <name>.json.
    """
    meta_path = cache_path.with_suffix(".json")
    texts = [dense_text(c) for c in chunks]
    meta = {
        "model": model_name,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "texts_sha256": hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest(),
    }
    cached = meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) == meta
    if cached and cache_path.exists():
        return np.load(cache_path)

    vectors = np.asarray(embed_passages(texts), dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return vectors


class DenseIndex:
    """One Qdrant collection per chunk configuration; point payload = the chunk dict."""

    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    @classmethod
    def build(
        cls, client: QdrantClient, collection: str, chunks: list[dict], vectors: np.ndarray
    ) -> "DenseIndex":
        """(Re)create the collection unless it already holds exactly these chunks."""
        if client.collection_exists(collection):
            if client.count(collection, exact=True).count == len(chunks):
                return cls(client, collection)
            client.delete_collection(collection)

        client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        batch = 256
        for start in range(0, len(chunks), batch):
            client.upsert(
                collection,
                points=[
                    models.PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
                    for i in range(start, min(start + batch, len(chunks)))
                ],
            )
        return cls(client, collection)

    def search(
        self, query_vector: np.ndarray, k: int, include_appendix: bool = True
    ) -> list[tuple[dict, float]]:
        """Top-k chunks by cosine similarity, best first; optionally without appendix chunks."""
        query_filter = None
        if not include_appendix:
            not_appendix = models.FieldCondition(
                key="is_appendix", match=models.MatchValue(value=False)
            )
            query_filter = models.Filter(must=[not_appendix])
        response = self.client.query_points(
            self.collection,
            query=np.asarray(query_vector).tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [(point.payload, point.score) for point in response.points]

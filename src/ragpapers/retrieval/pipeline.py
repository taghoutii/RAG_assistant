"""Phase 4 retrieval pipeline: BM25 + dense -> RRF fusion -> cross-encoder rerank.

Every stage is timed separately, so latency can be reported per stage:
    bm25    BM25 scoring of the query against every chunk, top `fusion_depth`
    dense   query embedding + Qdrant search, top `fusion_depth`
    fusion  Reciprocal Rank Fusion of the two rankings (retrieval/hybrid.py)
    rerank  cross-encoder over the top `rerank_depth` fused chunks (retrieval/rerank.py)
"""

import time

from qdrant_client import QdrantClient

from ragpapers.chunking.build import chunks_path, config_name, load_chunks
from ragpapers.config import REPO_ROOT
from ragpapers.index.bm25 import BM25Index
from ragpapers.index.dense import DenseIndex, Embedder, load_or_compute_embeddings
from ragpapers.retrieval.hybrid import reciprocal_rank_fusion


def elapsed_ms(start: float) -> float:
    return 1000 * (time.perf_counter() - start)


def load_indexes(config: dict) -> dict:
    """Chunks, BM25 and dense indexes for the configured chunk set (embeddings from cache).

    Returns a dict with "chunks", "by_id", "name", "bm25", "dense", "embedder" and "client"
    (close the client when done: it holds the local Qdrant lock).
    """
    emb = config["embedding"]
    name = config_name(config["chunking"])
    chunks = load_chunks(chunks_path(config))
    embedder = Embedder(emb["model"], emb["batch_size"])
    cache = REPO_ROOT / config["index"]["embeddings_dir"] / f"{name}.npy"
    vectors = load_or_compute_embeddings(chunks, embedder.encode_passages, emb["model"], cache)
    client = QdrantClient(path=str(REPO_ROOT / config["index"]["qdrant_path"]))
    return {
        "chunks": chunks,
        "by_id": {c["chunk_id"]: c for c in chunks},
        "name": name,
        "bm25": BM25Index(chunks, **config["bm25"]),
        "dense": DenseIndex.build(client, f"chunks_{name}", chunks, vectors),
        "embedder": embedder,
        "client": client,
    }


def first_stage(
    query: str, indexes: dict, fusion_depth: int, rrf_k: int
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Rankings of chunk IDs {"bm25", "dense", "hybrid"} and stage latencies in ms.

    "hybrid" is the full RRF fusion of BM25 top-`fusion_depth` and dense top-`fusion_depth`
    (up to 2 x fusion_depth IDs, best first).
    """
    rankings, ms = {}, {}

    start = time.perf_counter()
    rankings["bm25"] = [c["chunk_id"] for c, _ in indexes["bm25"].search(query, fusion_depth)]
    ms["bm25"] = elapsed_ms(start)

    start = time.perf_counter()
    vector = indexes["embedder"].encode_query(query)
    rankings["dense"] = [c["chunk_id"] for c, _ in indexes["dense"].search(vector, fusion_depth)]
    ms["dense"] = elapsed_ms(start)

    start = time.perf_counter()
    fused = reciprocal_rank_fusion([rankings["bm25"], rankings["dense"]], rrf_k=rrf_k)
    rankings["hybrid"] = [chunk_id for chunk_id, _ in fused]
    ms["fusion"] = elapsed_ms(start)

    return rankings, ms


def rerank_top(
    query: str, hybrid_ids: list[str], rerank_depth: int, by_id: dict, reranker
) -> tuple[list[str], float]:
    """The top `rerank_depth` fused IDs reordered by the cross-encoder, and the time it took."""
    start = time.perf_counter()
    ranked = reranker.rerank(query, [by_id[i] for i in hybrid_ids[:rerank_depth]])
    return [c["chunk_id"] for c, _ in ranked], elapsed_ms(start)

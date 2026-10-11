"""BM25, dense (in-memory Qdrant, hand-made vectors) and the embedding cache. No model needed."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from ragpapers.chunking.build import bm25_text, dense_text
from ragpapers.index.bm25 import BM25Index, tokenize
from ragpapers.index.dense import DenseIndex, load_or_compute_embeddings


def chunk(cid, text, section="1 Intro", appendix=False):
    return {
        "chunk_id": cid,
        "paper_id": "p",
        "title": "Paper",
        "year": 2020,
        "section": section,
        "section_url": "u",
        "source_url": "s",
        "is_appendix": appendix,
        "text": text,
        "n_words": len(text.split()),
        "dense_header": f"Paper > {section}",
    }


CHUNKS = [
    chunk("p:0000", "LoRA freezes the pretrained weights and trains low-rank matrices."),
    chunk("p:0001", "The decoder attends over all encoded passages at once."),
    chunk("p:0002", "Hyperparameters: learning rate and batch size per task.", "A.1 Setup", True),
]


def test_index_texts():
    assert bm25_text(CHUNKS[2]) == "Paper > A.1 Setup\n" + CHUNKS[2]["text"]
    assert dense_text(CHUNKS[2]) == "Paper > A.1 Setup\n" + CHUNKS[2]["text"]


# --- BM25 --------------------------------------------------------------------------------------


def test_tokenize_lowercases_splits_and_drops_stopwords():
    assert tokenize("How does Self-Attention work in BERT?") == [
        "self",
        "attention",
        "work",
        "bert",
    ]


def test_bm25_ranks_the_matching_chunk_first():
    index = BM25Index(CHUNKS)
    results = index.search("which weights does LoRA freeze?", k=2)
    assert results[0][0]["chunk_id"] == "p:0000"
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_bm25_searches_the_section_header_too():
    # "Setup" only appears in the section path, which is part of the indexed text.
    assert BM25Index(CHUNKS).search("setup", k=1)[0][0]["chunk_id"] == "p:0002"


# --- dense -------------------------------------------------------------------------------------

VECTORS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


@pytest.fixture()
def dense():
    client = QdrantClient(":memory:")
    return DenseIndex.build(client, "test", CHUNKS, VECTORS)


def test_dense_returns_nearest_chunk_with_payload(dense):
    results = dense.search(np.array([0.1, 0.9, 0.0]), k=2)
    assert [c["chunk_id"] for c, _ in results] == ["p:0001", "p:0000"]
    assert results[0][0]["section"] == "1 Intro"


def test_dense_can_exclude_appendix_chunks(dense):
    query = np.array([0.0, 0.0, 1.0])  # closest to the appendix chunk
    assert dense.search(query, k=1)[0][0]["chunk_id"] == "p:0002"
    without = dense.search(query, k=3, include_appendix=False)
    assert all(not c["is_appendix"] for c, _ in without)
    assert len(without) == 2


def test_dense_build_reuses_a_complete_collection(dense):
    again = DenseIndex.build(dense.client, "test", CHUNKS, VECTORS)
    assert again.client.count("test", exact=True).count == 3


# --- embedding cache ---------------------------------------------------------------------------


def test_embeddings_are_cached_and_recomputed_when_chunks_change(tmp_path):
    calls = []

    def fake_embed(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    cache = tmp_path / "emb.npy"
    first = load_or_compute_embeddings(CHUNKS, fake_embed, "m", cache)
    second = load_or_compute_embeddings(CHUNKS, fake_embed, "m", cache)
    assert calls == [3]  # second call came from the cache
    assert np.array_equal(first, second)

    load_or_compute_embeddings(CHUNKS[:2], fake_embed, "m", cache)  # different chunk ids
    load_or_compute_embeddings(CHUNKS[:2], fake_embed, "other-model", cache)  # different model
    changed = [dict(c, dense_header="New header") for c in CHUNKS[:2]]  # same ids, other text
    load_or_compute_embeddings(changed, fake_embed, "other-model", cache)
    assert calls == [3, 2, 2, 2]

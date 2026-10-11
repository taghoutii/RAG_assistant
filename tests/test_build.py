"""Chunk building: headers and their cap, the token budget check, quotes cut, config validation.

Uses a fake counter (one token per word), so no tokenizer download is needed.
"""

import copy

import pytest

from ragpapers.chunking.build import (
    bm25_text,
    build_chunks,
    config_name,
    dense_text,
    make_dense_header,
    over_budget,
    quotes_cut,
    short_title,
    validate_chunking_config,
)
from ragpapers.chunking.chunker import word_count
from ragpapers.config import load_config

HEADER = {"format": "{short_title} > {section}", "max_tokens": 6}


def test_config_name_keeps_the_phase3_word_names():
    assert config_name("tokens", 256, 64) == "tokens256_ov64"
    assert config_name("words", 150, 30) == "size150_ov30"


def test_short_title_is_the_part_before_the_colon():
    assert short_title("LoRA: Low-Rank Adaptation of Large Language Models") == "LoRA"
    assert short_title("Attention Is All You Need") == "Attention Is All You Need"


def test_dense_header_uses_the_last_heading():
    header = make_dense_header(
        "BERT: Pre-training", "3 BERT > 3.1 Pre-training BERT", HEADER["format"], 10, word_count
    )
    assert header == "BERT > 3.1 Pre-training BERT"


def test_dense_header_is_capped_at_word_boundaries():
    section = "D.1 Clustering > ArxivClusteringS2S, ArxivClusteringP2P, BiorxivClusteringS2S, More"
    header = make_dense_header("MTEB: Massive", section, HEADER["format"], 4, word_count)
    assert header == "MTEB > ArxivClusteringS2S, …"
    assert word_count(header) <= 4


def chunk(text, section="1 Intro", title="T: Long title"):
    return {
        "chunk_id": "p:0000",
        "paper_id": "p",
        "title": title,
        "section": section,
        "text": text,
        "dense_header": make_dense_header(title, section, HEADER["format"], 6, word_count),
    }


def test_bm25_gets_the_full_header_and_dense_the_short_one():
    c = chunk("body text", section="2 Method > 2.1 Details")
    assert bm25_text(c) == "T: Long title > 2 Method > 2.1 Details\nbody text"
    assert dense_text(c) == "T > 2.1 Details\nbody text"


def test_over_budget_counts_header_text_and_special_tokens():
    fits = chunk("w " * 6)  # header "T > 1 Intro" = 4 + 6 words + 2 special = 12
    too_long = dict(fits, chunk_id="p:0001", text="w " * 7)  # 13
    assert over_budget([fits], word_count, special_tokens=2, max_sequence_tokens=12) == []
    assert over_budget([fits, too_long], word_count, 2, 12) == [("p:0001", 13)]


def test_quotes_cut_reports_quotes_no_single_chunk_contains():
    chunks = [
        dict(chunk("the first half of a sentence"), paper_id="p"),
        dict(chunk("and here is the second half"), paper_id="p"),
    ]
    questions = [
        {"id": "q1", "evidence": [{"paper_id": "p", "quote": "the first half of a sentence"}]},
        {"id": "q2", "evidence": [{"paper_id": "p", "quote": "a sentence and here is the second"}]},
    ]
    assert quotes_cut(chunks, questions, threshold=90) == ["q2/p"]


def test_build_chunks_adds_a_dense_header_to_every_chunk():
    paper = {
        "paper_id": "p",
        "title": "Paper: A Study",
        "year": 2020,
        "source_url": "s",
        "sections": [
            {
                "path": ["1 Intro"],
                "url": "u",
                "is_appendix": False,
                "blocks": [{"type": "paragraph", "text": "x y z"}],
            },
        ],
    }
    (c,) = build_chunks({"p": paper}, 10, 2, word_count, HEADER)
    assert c["dense_header"] == "Paper > 1 Intro"
    assert c["n_tokens"] == 3


# --- config validation -------------------------------------------------------------------------


@pytest.fixture()
def config():
    return copy.deepcopy(load_config())


def test_committed_config_is_valid(config):
    validate_chunking_config(config, special_tokens=2)


def test_chunk_size_over_the_budget_is_rejected(config):
    config["chunking"]["compare"] = [{"chunk_size": 479, "overlap": 64}]  # budget is 478
    with pytest.raises(ValueError, match="body budget of 478"):
        validate_chunking_config(config, special_tokens=2)


def test_tokenizer_must_match_the_embedding_model(config):
    config["chunking"]["tokenizer"] = "some/other-model"
    with pytest.raises(ValueError, match="must be the embedding model"):
        validate_chunking_config(config, special_tokens=2)


@pytest.mark.parametrize(
    ("key", "value"), [("unit", "characters"), ("compare", [{"chunk_size": 64, "overlap": 64}])]
)
def test_bad_unit_or_overlap_is_rejected(config, key, value):
    config["chunking"][key] = value
    with pytest.raises(ValueError):
        validate_chunking_config(config, special_tokens=2)

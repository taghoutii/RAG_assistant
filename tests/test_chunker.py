import math

import pytest

from ragpapers.chunking.chunker import (
    chunk_paper,
    chunk_section,
    overlap_tail,
    split_block,
    split_long_word,
    word_count,
)


def para(text):
    return {"type": "paragraph", "text": text}


def words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(1, n + 1))


def fake_tokens(text):
    """Fake tokenizer: each word costs ceil(len(word) / 3) tokens (additive, like bge's)."""
    return sum(math.ceil(len(w) / 3) for w in text.split())


# --- split_block (word counter: the original behaviour) ----------------------------------------


def test_block_that_fits_is_unchanged():
    assert split_block("one two three", "paragraph", 3, word_count) == ["one two three"]


def test_paragraph_is_split_on_sentences():
    text = "One two three. Four five six. Seven eight."
    assert split_block(text, "paragraph", 6, word_count) == [
        "One two three. Four five six.",
        "Seven eight.",
    ]


def test_decimal_points_are_not_sentence_ends():
    text = "It scores 28.4 BLEU here. Then 41.8 BLEU there."
    assert split_block(text, "paragraph", 5, word_count) == [
        "It scores 28.4 BLEU here.",
        "Then 41.8 BLEU there.",
    ]


def test_table_is_split_on_whole_rows():
    assert split_block("a b\nc d\ne f", "table", 4, word_count) == ["a b\nc d", "e f"]


def test_list_is_split_on_whole_items():
    # Items have no sentence punctuation: a sentence split would see one 11-word "sentence" and
    # cut it into word windows across item boundaries ("• a b c • d e ...").
    text = "• a b c\n• d e\n• f g h"
    assert split_block(text, "list", 7, word_count) == ["• a b c\n• d e", "• f g h"]


def test_overlong_sentence_falls_back_to_word_windows():
    assert split_block(words("w", 10), "paragraph", 4, word_count) == [
        "w1 w2 w3 w4",
        "w5 w6 w7 w8",
        "w9 w10",
    ]


def test_word_windows_are_packed_with_other_sentences():
    text = "Short one. " + words("x", 7) + "."
    assert split_block(text, "paragraph", 4, word_count) == [
        "Short one.",
        "x1 x2 x3 x4",
        "x5 x6 x7.",
    ]


def test_split_pieces_never_exceed_max_tokens():
    text = " ".join(f"Sentence {i} has a few words." for i in range(30)) + " " + words("z", 25)
    pieces = split_block(text, "paragraph", 12, word_count)
    assert all(word_count(p) <= 12 for p in pieces)
    assert " ".join(pieces).split() == text.split()  # nothing lost, nothing reordered


# --- split_block (token counter) ---------------------------------------------------------------


def test_sentences_are_packed_by_token_cost_not_word_count():
    # "Transformers" costs 4 tokens, so the first sentence (2 words) already costs 5.
    text = "Transformers work. Yes it does."
    assert fake_tokens("Transformers work.") == 6
    assert split_block(text, "paragraph", 6, fake_tokens) == ["Transformers work.", "Yes it does."]


def test_word_windows_are_measured_in_tokens():
    # Each 6-letter word costs 2 tokens: a 5-token window holds 2 words.
    text = "abcdef ghijkl mnopqr stuvwx"
    assert split_block(text, "paragraph", 5, fake_tokens) == ["abcdef ghijkl", "mnopqr stuvwx"]


def test_giant_word_is_sliced_inside_split_block():
    giant = "x" * 30  # 10 tokens on its own
    pieces = split_block(f"Intro. {giant}", "paragraph", 4, fake_tokens)
    assert all(fake_tokens(p) <= 4 for p in pieces)
    assert "".join(p.replace(" ", "") for p in pieces) == "Intro." + giant  # nothing lost


# --- split_long_word ---------------------------------------------------------------------------


def test_split_long_word_cuts_into_slices_within_the_limit():
    word = "abcdefghij"  # 10 characters -> 4 fake tokens
    assert split_long_word(word, 2, fake_tokens) == ["abcdef", "ghij"]


def test_split_long_word_keeps_every_character_in_order():
    word = "$\\boldsymbol{q}_{m}^{\\intercal}\\boldsymbol{k}_{n}$"
    slices = split_long_word(word, 3, fake_tokens)
    assert "".join(slices) == word
    assert all(1 <= fake_tokens(s) <= 3 for s in slices)


def test_split_long_word_leaves_a_word_that_fits():
    assert split_long_word("abcdef", 2, fake_tokens) == ["abcdef"]


# --- overlap_tail ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overlap", "expected"),
    [(3, "bbbbbb cc"), (2, "cc"), (1, "cc"), (0, "")],
)
def test_overlap_tail_takes_whole_words_up_to_the_budget(overlap, expected):
    # Fake costs: "aa" = 1, "bbbbbb" = 2, "cc" = 1.
    assert overlap_tail("aa bbbbbb cc", overlap, fake_tokens) == expected


def test_overlap_tail_is_empty_when_the_last_word_alone_is_too_long():
    assert overlap_tail("aa bbbbbbbbb", 2, fake_tokens) == ""  # last word costs 3


def test_overlap_tail_with_word_counter_is_the_last_n_words():
    assert overlap_tail("a1 a2 a3\nb1 b2", 2, word_count) == "b1 b2"
    assert overlap_tail("a1", 5, word_count) == "a1"  # shorter than the overlap


# --- chunk_section (word counter: the original behaviour) --------------------------------------


def test_worked_example_with_overlap():
    blocks = [para("a1 a2 a3"), para("b1 b2"), para("c1 c2 c3")]
    assert chunk_section(blocks, 6, 2, word_count) == [
        "a1 a2 a3\nb1 b2",
        "b1 b2\nc1 c2 c3",
    ]


def test_no_overlap():
    blocks = [para("a1 a2 a3"), para("b1 b2"), para("c1 c2 c3")]
    assert chunk_section(blocks, 6, 0, word_count) == ["a1 a2 a3\nb1 b2", "c1 c2 c3"]


def test_oversized_block_is_split_then_packed_with_overlap():
    blocks = [para(words("x", 10))]
    assert chunk_section(blocks, 6, 2, word_count) == [
        "x1 x2 x3 x4",
        "x3 x4\nx5 x6 x7 x8",
        "x7 x8\nx9 x10",
    ]


def test_equation_that_fits_is_never_cut():
    equation = {"type": "equation", "text": "$h=W_{0}x+BAx$ (3)"}
    blocks = [para(words("p", 4)), equation, para(words("q", 4))]
    chunks = chunk_section(blocks, 6, 1, word_count)
    assert any(equation["text"] in c for c in chunks)


def test_chunks_never_exceed_chunk_size():
    blocks = [para(" ".join(f"Sentence {i} is here." for i in range(40)))]
    blocks += [{"type": "table", "text": "\n".join(f"row {i} | {i * 2}" for i in range(50))}]
    chunks = chunk_section(blocks, 30, 5, word_count)
    assert len(chunks) > 1
    assert all(word_count(c) <= 30 for c in chunks)


def test_empty_section():
    assert chunk_section([], 10, 2, word_count) == []


@pytest.mark.parametrize("overlap", [-1, 6, 7])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_section([para("a b c")], 6, overlap, word_count)


# --- chunk_section (token counter) -------------------------------------------------------------


def test_overlap_is_counted_in_tokens_and_never_splits_a_word():
    # Costs: "aaa" = 1, "bbbbbb" = 2, "ccccccccc" = 3. Chunk budget 6 tokens, overlap 3 tokens.
    blocks = [para("aaa bbbbbb"), para("ccccccccc"), para("aaa aaa aaa")]
    chunks = chunk_section(blocks, 6, 3, fake_tokens)
    assert chunks == ["aaa bbbbbb\nccccccccc", "ccccccccc\naaa aaa aaa"]
    assert all(fake_tokens(c) <= 6 for c in chunks)


def test_token_chunks_never_exceed_chunk_size_even_with_giant_words():
    blocks = [para(" ".join(f"Sentence number {i} is here." for i in range(40)))]
    blocks += [{"type": "equation", "text": "$" + "x" * 90 + "$"}]  # one 31-token "word"
    blocks += [{"type": "table", "text": "\n".join(f"row {i} | {i * 2}" for i in range(50))}]
    chunks = chunk_section(blocks, 20, 5, fake_tokens)
    assert len(chunks) > 1
    assert all(fake_tokens(c) <= 20 for c in chunks)


# --- chunk_paper -------------------------------------------------------------------------------

PAPER = {
    "paper_id": "1234.56789",
    "title": "A Paper",
    "year": 2020,
    "source_url": "https://arxiv.org/abs/1234.56789v1",
    "sections": [
        {
            "section_id": "S1",
            "path": ["1 Intro"],
            "is_appendix": False,
            "url": "https://arxiv.org/html/1234.56789v1#S1",
            "blocks": [para(words("i", 5)), para(words("j", 5))],
        },
        {
            "section_id": "A1",
            "path": ["Appendix A Extra", "A.1 More"],
            "is_appendix": True,
            "url": "https://arxiv.org/html/1234.56789v1#A1",
            "blocks": [para(words("k", 3))],
        },
    ],
}


def test_chunk_paper_metadata_and_ids():
    chunks = chunk_paper(PAPER, 6, 1, word_count)
    assert [c["chunk_id"] for c in chunks] == [
        "1234.56789:0000",
        "1234.56789:0001",
        "1234.56789:0002",
    ]
    last = chunks[-1]
    assert last == {
        "chunk_id": "1234.56789:0002",
        "paper_id": "1234.56789",
        "title": "A Paper",
        "year": 2020,
        "section": "Appendix A Extra > A.1 More",
        "section_url": "https://arxiv.org/html/1234.56789v1#A1",
        "source_url": "https://arxiv.org/abs/1234.56789v1",
        "is_appendix": True,
        "text": "k1 k2 k3",
        "n_words": 3,
        "n_tokens": 3,
    }


def test_chunk_paper_counts_tokens_with_the_given_counter():
    chunks = chunk_paper(PAPER, 6, 1, fake_tokens)
    assert all(c["n_tokens"] == fake_tokens(c["text"]) for c in chunks)
    assert all(c["n_tokens"] <= 6 for c in chunks)


def test_chunks_do_not_cross_sections():
    chunks = chunk_paper(PAPER, 6, 1, word_count)
    appendix_chunk = chunks[-1]["text"]
    assert "j" not in appendix_chunk  # no overlap carried from the previous section


def test_appendix_can_be_excluded():
    chunks = chunk_paper(PAPER, 6, 1, word_count, include_appendix=False)
    assert all(not c["is_appendix"] for c in chunks)
    assert len(chunks) == 2

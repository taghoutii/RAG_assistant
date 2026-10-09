import pytest

from ragpapers.chunking.chunker import chunk_paper, chunk_section, split_block, word_count


def para(text):
    return {"type": "paragraph", "text": text}


def words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(1, n + 1))


# --- split_block -------------------------------------------------------------------------------


def test_block_that_fits_is_unchanged():
    assert split_block("one two three", "paragraph", max_words=3) == ["one two three"]


def test_paragraph_is_split_on_sentences():
    text = "One two three. Four five six. Seven eight."
    assert split_block(text, "paragraph", max_words=6) == [
        "One two three. Four five six.",
        "Seven eight.",
    ]


def test_decimal_points_are_not_sentence_ends():
    text = "It scores 28.4 BLEU here. Then 41.8 BLEU there."
    assert split_block(text, "paragraph", max_words=5) == [
        "It scores 28.4 BLEU here.",
        "Then 41.8 BLEU there.",
    ]


def test_table_is_split_on_whole_rows():
    assert split_block("a b\nc d\ne f", "table", max_words=4) == ["a b\nc d", "e f"]


def test_list_is_split_on_whole_items():
    # Items have no sentence punctuation: a sentence split would see one 11-word "sentence" and
    # cut it into word windows across item boundaries ("• a b c • d e ...").
    text = "• a b c\n• d e\n• f g h"
    assert split_block(text, "list", max_words=7) == ["• a b c\n• d e", "• f g h"]


def test_overlong_sentence_falls_back_to_word_windows():
    assert split_block(words("w", 10), "paragraph", max_words=4) == [
        "w1 w2 w3 w4",
        "w5 w6 w7 w8",
        "w9 w10",
    ]


def test_word_windows_are_packed_with_other_sentences():
    text = "Short one. " + words("x", 7) + "."
    assert split_block(text, "paragraph", max_words=4) == ["Short one.", "x1 x2 x3 x4", "x5 x6 x7."]


def test_split_pieces_never_exceed_max_words():
    text = " ".join(f"Sentence {i} has a few words." for i in range(30)) + " " + words("z", 25)
    pieces = split_block(text, "paragraph", max_words=12)
    assert all(word_count(p) <= 12 for p in pieces)
    assert " ".join(pieces).split() == text.split()  # nothing lost, nothing reordered


# --- chunk_section -----------------------------------------------------------------------------


def test_worked_example_with_overlap():
    blocks = [para("a1 a2 a3"), para("b1 b2"), para("c1 c2 c3")]
    assert chunk_section(blocks, chunk_size=6, overlap=2) == [
        "a1 a2 a3\nb1 b2",
        "b1 b2\nc1 c2 c3",
    ]


def test_no_overlap():
    blocks = [para("a1 a2 a3"), para("b1 b2"), para("c1 c2 c3")]
    assert chunk_section(blocks, chunk_size=6, overlap=0) == ["a1 a2 a3\nb1 b2", "c1 c2 c3"]


def test_oversized_block_is_split_then_packed_with_overlap():
    blocks = [para(words("x", 10))]
    assert chunk_section(blocks, chunk_size=6, overlap=2) == [
        "x1 x2 x3 x4",
        "x3 x4\nx5 x6 x7 x8",
        "x7 x8\nx9 x10",
    ]


def test_equation_that_fits_is_never_cut():
    equation = {"type": "equation", "text": "$h=W_{0}x+BAx$ (3)"}
    blocks = [para(words("p", 4)), equation, para(words("q", 4))]
    chunks = chunk_section(blocks, chunk_size=6, overlap=1)
    assert any(equation["text"] in c for c in chunks)


def test_chunks_never_exceed_chunk_size():
    blocks = [para(" ".join(f"Sentence {i} is here." for i in range(40)))]
    blocks += [{"type": "table", "text": "\n".join(f"row {i} | {i * 2}" for i in range(50))}]
    chunks = chunk_section(blocks, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    assert all(word_count(c) <= 30 for c in chunks)


def test_empty_section():
    assert chunk_section([], chunk_size=10, overlap=2) == []


@pytest.mark.parametrize("overlap", [-1, 6, 7])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_section([para("a b c")], chunk_size=6, overlap=overlap)


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
    chunks = chunk_paper(PAPER, chunk_size=6, overlap=1)
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
    }


def test_chunks_do_not_cross_sections():
    chunks = chunk_paper(PAPER, chunk_size=6, overlap=1)
    appendix_chunk = chunks[-1]["text"]
    assert "j" not in appendix_chunk  # no overlap carried from the previous section


def test_appendix_can_be_excluded():
    chunks = chunk_paper(PAPER, chunk_size=6, overlap=1, include_appendix=False)
    assert all(not c["is_appendix"] for c in chunks)
    assert len(chunks) == 2

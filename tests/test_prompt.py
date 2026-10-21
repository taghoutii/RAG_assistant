import pytest

from ragpapers.config import load_config
from ragpapers.generation.prompt import (
    SYSTEM_TEMPLATE,
    USER_TEMPLATE,
    Citations,
    build_messages,
    format_context,
    is_abstention,
    parse_citations,
)

ABSTAIN = load_config()["abstention_text"]
QUESTION = "How often does REALM rebuild its document index while the retriever is being trained?"


def chunk(n, title="REALM: Retrieval-Augmented Language Model Pre-Training", section=None):
    return {
        "chunk_id": f"2002.08909:{n:04d}",
        "title": title,
        "section": section or f"3 Approach > Section {n}",
        "text": f"Text of chunk number {n}, with its own details.",
    }


CHUNKS = [chunk(n) for n in range(1, 9)]  # context_k = 8


# --- templates ----------------------------------------------------------------------------------


def test_templates_have_their_placeholders():
    assert "{abstention_text}" in SYSTEM_TEMPLATE
    assert "{context}" in USER_TEMPLATE and "{question}" in USER_TEMPLATE
    assert USER_TEMPLATE.index("{context}") < USER_TEMPLATE.index("{question}")


# --- format_context -----------------------------------------------------------------------------


def test_context_numbers_chunks_from_1_in_order_with_their_source():
    context = format_context(CHUNKS)
    positions = [context.index(f"[{n}] ") for n in range(1, 9)]
    assert positions == sorted(positions) and context.startswith("[1] ")
    for n, c in enumerate(CHUNKS, start=1):
        assert f"[{n}] {c['title']} > {c['section']}\n{c['text']}" in context
    assert "[9]" not in context and "[0]" not in context


def test_context_separates_chunks_with_a_blank_line():
    context = format_context(CHUNKS[:2])
    assert f"{CHUNKS[0]['text']}\n\n[2] " in context


# --- build_messages -----------------------------------------------------------------------------


def test_messages_are_system_then_user():
    messages = build_messages(QUESTION, CHUNKS, ABSTAIN)
    assert [m.role for m in messages] == ["system", "user"]


def test_system_message_has_the_exact_abstention_text_from_the_config():
    (system, _) = build_messages(QUESTION, CHUNKS, "No evidence, sorry.")
    assert "No evidence, sorry." in system.content
    assert "{" not in system.content  # every placeholder filled


def test_user_message_has_the_chunks_then_the_question():
    (system, user) = build_messages(QUESTION, CHUNKS, ABSTAIN)
    assert format_context(CHUNKS) in user.content
    assert user.content.index(CHUNKS[-1]["text"]) < user.content.index(QUESTION)
    assert CHUNKS[0]["text"] not in system.content  # paper text is data, not instructions


def test_messages_are_deterministic():
    assert build_messages(QUESTION, CHUNKS, ABSTAIN) == build_messages(QUESTION, CHUNKS, ABSTAIN)


def test_chunk_text_with_braces_is_kept_verbatim():
    tricky = [chunk(1) | {"text": "Loss L = sum_{i} x_{i} over {question} and {context}."}]
    (_, user) = build_messages(QUESTION, tricky, ABSTAIN)
    assert tricky[0]["text"] in user.content and QUESTION in user.content


@pytest.mark.parametrize(("question", "chunks"), [(QUESTION, []), ("   ", CHUNKS)])
def test_no_chunks_or_blank_question_is_rejected(question, chunks):
    with pytest.raises(ValueError):
        build_messages(question, chunks, ABSTAIN)


# --- parse_citations ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("REALM refreshes the index about every 500 steps [1].", [1]),
        ("Claim one [2]. Claim two [5]. Claim one again [2].", [2, 5]),
        ("Both papers agree [3][1].", [3, 1]),
        ("Both papers agree [4, 2] and [4,7].", [4, 2, 7]),
        ("No citation at all.", []),
        ("Not citations: [a], [3a], (3), [], [ ].", []),
    ],
)
def test_parse_citations(answer, expected):
    assert parse_citations(answer, 8) == Citations(valid=expected, invalid=[])


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("REALM refreshes the index about every 500 steps【1】.", [1]),  # 【1】
        ("Both papers agree 【2, 5】.", [2, 5]),  # 【2, 5】
        ("Both papers agree 【2,5】【3】.", [2, 5, 3]),  # 【2,5】【3】
        ("Claim [4]. Claim 【1】. Claim [1][6].", [4, 1, 6]),  # mixed, repeat once
        ("Not citations: 【a】, 【3a】, 【】.", []),
    ],
)
def test_parse_lenticular_and_mixed_citations(answer, expected):
    """gpt-oss sometimes cites with its native 【n】 brackets instead of [n]."""
    assert parse_citations(answer, 8) == Citations(valid=expected, invalid=[])


def test_out_of_range_lenticular_citations_are_invalid():
    assert parse_citations("Fact 【9】. Fact [9]. Fact 【0, 2】.", 8) == Citations(
        valid=[2], invalid=[9, 0]
    )


def test_out_of_range_citations_are_invalid():
    assert parse_citations("Fact [0]. Fact [9]. Fact [8]. Fact [9].", 8) == Citations(
        valid=[8], invalid=[0, 9]
    )


# --- is_abstention ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        ABSTAIN,
        f"  {ABSTAIN}\n",
        f'"{ABSTAIN}"',
        ABSTAIN.replace("'", "’"),  # curly apostrophe
        ABSTAIN.replace(" ", "\n", 1),
    ],
)
def test_abstention_is_detected(answer):
    assert is_abstention(answer, ABSTAIN)


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "REALM refreshes the index about every 500 steps [1].",
        f"{ABSTAIN} However, chunk [2] mentions asynchronous refreshes.",
        f"{ABSTAIN} [1]",
        "I could not find sufficient evidence in the available sources.",  # reworded
        ABSTAIN.rstrip("."),
    ],
)
def test_other_answers_are_not_abstentions(answer):
    assert not is_abstention(answer, ABSTAIN)

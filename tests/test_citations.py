from ragpapers.config import load_config
from ragpapers.generation.citations import (
    CitedAnswer,
    Source,
    cite_answer,
    format_source,
    format_sources,
)

ABSTAIN = load_config()["abstention_text"]
TITLE = "REALM: Retrieval-Augmented Language Model Pre-Training"


def chunk(n, section):
    return {
        "chunk_id": f"2002.08909:{n:04d}",
        "paper_id": "2002.08909",
        "title": TITLE,
        "year": 2020,
        "section": section,
        "section_url": f"https://arxiv.org/html/2002.08909v1#S{n}",
        "text": f"chunk {n}",
    }


CHUNKS = [chunk(n, f"{n} Section {n}") for n in range(1, 9)]


def test_cited_chunks_become_sources_in_first_citation_order():
    cited = cite_answer("Refresh every 500 steps [5]. Asynchronous [2][5].", CHUNKS, ABSTAIN)
    assert not cited.abstained and cited.invalid_citations == []
    assert [s.number for s in cited.sources] == [5, 2]
    assert cited.sources[0] == Source(
        number=5,
        chunk_id="2002.08909:0005",
        paper_id="2002.08909",
        title=TITLE,
        year=2020,
        section="5 Section 5",
        url="https://arxiv.org/html/2002.08909v1#S5",
    )


def test_answer_text_is_kept_with_the_models_numbers():
    reply = "  Refresh every 500 steps [5].\n"
    assert cite_answer(reply, CHUNKS, ABSTAIN).text == "Refresh every 500 steps [5]."


def test_invalid_numbers_are_reported_not_listed():
    cited = cite_answer("A [9]. B [1]. C [0].", CHUNKS, ABSTAIN)
    assert [s.number for s in cited.sources] == [1]
    assert cited.invalid_citations == [9, 0]


def test_abstention_has_no_sources_and_the_exact_sentence():
    cited = cite_answer(f'"{ABSTAIN.replace(chr(39), chr(0x2019))}"\n', CHUNKS, ABSTAIN)
    assert cited == CitedAnswer(text=ABSTAIN, abstained=True)


def test_hedged_abstention_is_an_answer():
    cited = cite_answer(f"{ABSTAIN} But chunk [3] is related.", CHUNKS, ABSTAIN)
    assert not cited.abstained and [s.number for s in cited.sources] == [3]


def test_uncited_answer_has_no_sources():
    cited = cite_answer("REALM refreshes its index.", CHUNKS, ABSTAIN)
    assert not cited.abstained and cited.sources == []


def test_source_line_has_title_year_section_and_link():
    (source,) = cite_answer("Fact [2].", CHUNKS, ABSTAIN).sources
    assert format_source(source) == (
        f"[2] {TITLE} (2020), 2 Section 2. https://arxiv.org/html/2002.08909v1#S2"
    )


def test_sources_are_one_per_line():
    sources = cite_answer("A [3]. B [1].", CHUNKS, ABSTAIN).sources
    lines = format_sources(sources).splitlines()
    assert [line[:3] for line in lines] == ["[3]", "[1]"]
    assert format_sources([]) == ""

from pathlib import Path

import pytest

from ragpapers.ingest.html_parser import parse_latexml_html

FIXTURE = Path(__file__).parent / "fixtures" / "latexml_sample.html"


@pytest.fixture(scope="module")
def sections():
    return parse_latexml_html(FIXTURE.read_text(encoding="utf-8"))


def by_id(sections, section_id):
    return next(s for s in sections if s["section_id"] == section_id)


def all_text(sections):
    return "\n".join(b["text"] for s in sections for b in s["blocks"])


def test_sections_in_document_order_with_paths(sections):
    assert [s["section_id"] for s in sections] == ["abstract", "S1", "S1.SS1", "A1"]
    assert by_id(sections, "S1.SS1")["path"] == ["1 Introduction", "1.1 Results"]


def test_abstract_is_first(sections):
    assert sections[0]["blocks"] == [{"type": "paragraph", "text": "We study sample parsing."}]


def test_noise_is_removed(sections):
    text = all_text(sections)
    for unwanted in ["Navigation", "Jane Doe", "In Proceedings of", "We thank our funders"]:
        assert unwanted not in text


def test_math_citations_and_footnotes(sections):
    paragraph = by_id(sections, "S1")["blocks"][0]["text"]
    assert paragraph == (
        "Transformers [ref] use keys of size $d_{k}$ as shown by Devlin et al. (2019). "
        "Results follow. [Footnote: Code is public.]"
    )


def test_equation_block_strips_displaystyle(sections):
    equation = by_id(sections, "S1")["blocks"][1]
    assert equation == {"type": "equation", "text": "$y=Wx$ (1)"}


def test_table_keeps_caption_and_rows(sections):
    table = by_id(sections, "S1.SS1")["blocks"][0]
    assert table["type"] == "table"
    assert table["text"] == "Table 1: BLEU scores.\nModel | BLEU\nBase | 27.3"


def test_figure_keeps_caption_only(sections):
    figure = by_id(sections, "S1.SS1")["blocks"][1]
    assert figure == {"type": "figure", "text": "Figure 1: Model overview."}


def test_nested_table_panels_become_separate_blocks(sections):
    appendix = by_id(sections, "A1")
    assert appendix["is_appendix"] is True
    types_and_heads = [(b["type"], b["text"].split("\n")[0]) for b in appendix["blocks"]]
    assert types_and_heads[:3] == [
        ("figure", "Table 2: Example outputs."),
        ("table", "Table 3: First panel."),
        ("table", "Table 4: Second panel."),
    ]


def test_leaked_footnotetext_becomes_footnote(sections):
    texts = [b["text"] for b in by_id(sections, "A1")["blocks"]]
    assert "[Footnote: See the FAQ.]" in texts


def test_list_items_one_per_line(sections):
    lst = by_id(sections, "A1")["blocks"][-1]
    assert lst == {"type": "list", "text": "• first item\n• second item"}


def test_non_latexml_input_is_rejected():
    with pytest.raises(ValueError):
        parse_latexml_html("<html><body><p>plain page</p></body></html>")

"""Parse arXiv LaTeXML HTML (arxiv.org/html) into sections made of typed text blocks.

Output: a list of sections in document order. Nested sections are flattened, and each one keeps
its full heading path:

    {
        "section_id": "S3.SS2",            # HTML anchor, used to link straight to the section
        "heading": "3.2 Attention",
        "path": ["3 Model Architecture", "3.2 Attention"],
        "is_appendix": False,
        "blocks": [{"type": "paragraph", "text": "..."}, ...],
    }

Block types: paragraph, equation, table, figure (caption only), list, code.

Cleaning rules:
- math is replaced by its LaTeX source, written as $...$
- numeric citations ("[13]", "[3, 7]") are replaced by "[ref]"; author-year citations are kept
- footnotes are moved to the end of their paragraph as "[Footnote: ...]"
- bibliography, acknowledgements, author block, images and page chrome are dropped
"""

import re

from bs4 import BeautifulSoup, NavigableString, Tag

from ragpapers.ingest.clean import is_numeric_citation, normalize_text

SECTION_CLASSES = {
    "ltx_part",
    "ltx_chapter",
    "ltx_section",
    "ltx_subsection",
    "ltx_subsubsection",
    "ltx_paragraph",
    "ltx_appendix",
}
BLOCK_TAGS = {"p", "div", "figure", "table", "ul", "ol", "dl", "pre", "blockquote", "section"}
NOISE_SELECTORS = [
    "nav",
    "header",
    "footer",
    "section.ltx_bibliography",
    "div.ltx_authors",
    "h1.ltx_title_document",
    "div.ltx_acknowledgements",
    "div.ltx_page_logo",
    "div.ltx_pagination",
    "div.ltx_dates",
    "div.ltx_classification",
    "div.ltx_keywords",
    "span.ltx_ERROR",  # LaTeX macros LaTeXML could not convert
    "img",
    "svg",
    "object",
]
CITATION_PLACEHOLDER = "[ref]"
# LaTeXML sometimes renders \footnotetext as a paragraph like "99footnotetext: See ...".
LEAKED_FOOTNOTETEXT = re.compile(r"^\S{0,4}footnotetext:\s*(.*)$", re.DOTALL)
# Sections with no answerable content: thanks, author lists, venue checklists.
SKIPPED_SECTION_TITLE = re.compile(
    r"\b(acknowledge?ments?|author list|author contributions|checklist)\b", re.IGNORECASE
)


def parse_latexml_html(html: str) -> list[dict]:
    """Turn one LaTeXML paper into a flat list of sections (see module docstring)."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_="ltx_document")
    if article is None:
        raise ValueError("Not a LaTeXML document (no <article class='ltx_document'>)")

    _remove_noise(article)
    _space_after_tags(article)
    _replace_math_with_latex(article)
    _replace_numeric_citations(article)

    sections: list[dict] = []
    abstract = article.find("div", class_="ltx_abstract")
    if abstract is not None:
        for title in abstract.find_all(class_="ltx_title_abstract"):
            title.decompose()
        sections.append(_new_section("abstract", "Abstract", ["Abstract"], False))
        sections[-1]["blocks"] = _blocks_from(abstract)
        abstract.decompose()

    # Content before the first numbered section (rare) goes into a "Front matter" section.
    front = _new_section("front", "Front matter", ["Front matter"], False)
    sections.append(front)
    for child in article.find_all(recursive=False):
        if _is_section(child):
            _collect_section(child, [], False, sections)
        elif not _is_heading(child):
            front["blocks"].extend(_blocks_from(child))

    return [s for s in sections if s["blocks"]]


# --- tree preparation --------------------------------------------------------------------------


def _remove_noise(article: Tag) -> None:
    for selector in NOISE_SELECTORS:
        for el in article.select(selector):
            el.decompose()


def _space_after_tags(article: Tag) -> None:
    """Numbering tags ("3.2", "•", "Table 1:") are sometimes glued to the text that follows."""
    for tag in article.select("span.ltx_tag"):
        tag.append(" ")


def _replace_math_with_latex(article: Tag) -> None:
    for math in article.find_all("math"):
        latex = (math.get("alttext") or "").replace("\\displaystyle", "").strip()
        math.replace_with(NavigableString(f"${latex}$" if latex else ""))


def _replace_numeric_citations(article: Tag) -> None:
    """Replace numeric citations like [13] with "[ref]".

    The number means nothing without the bibliography, but deleting the citation leaves broken
    sentences ("models such as and."). "[ref]" keeps them readable and marks the attribution.
    """
    for cite in article.find_all("cite"):
        if is_numeric_citation(cite.get_text("")):
            cite.replace_with(NavigableString(CITATION_PLACEHOLDER))


# --- sections ----------------------------------------------------------------------------------


def _is_section(el: Tag) -> bool:
    return el.name == "section" and bool(SECTION_CLASSES & set(el.get("class", [])))


def _is_heading(el: Tag) -> bool:
    return el.name in {"h1", "h2", "h3", "h4", "h5", "h6"} and "ltx_title" in el.get("class", [])


def _new_section(section_id: str, heading: str, path: list[str], is_appendix: bool) -> dict:
    return {
        "section_id": section_id,
        "heading": heading,
        "path": path,
        "is_appendix": is_appendix,
        "blocks": [],
    }


def _collect_section(el: Tag, parent_path: list[str], parent_is_appendix: bool, out: list) -> None:
    """Append this section, then its subsections, to `out` (document order)."""
    title = next((c for c in el.find_all(recursive=False) if _is_heading(c)), None)
    heading = _inline_text(title) if title is not None else ""
    if SKIPPED_SECTION_TITLE.search(heading):
        return

    is_appendix = parent_is_appendix or "ltx_appendix" in el.get("class", [])
    path = parent_path + [heading] if heading else parent_path
    section = _new_section(
        el.get("id", ""), heading or (path[-1] if path else ""), path, is_appendix
    )
    out.append(section)

    for child in el.find_all(recursive=False):
        if _is_section(child):
            _collect_section(child, path, is_appendix, out)
        elif not _is_heading(child):
            section["blocks"].extend(_blocks_from(child))


# --- blocks ------------------------------------------------------------------------------------


def _block(block_type: str, text: str) -> dict:
    return {"type": block_type, "text": text}


def _blocks_from(el: Tag) -> list[dict]:
    """Convert one element into zero or more text blocks."""
    if not isinstance(el, Tag) or _is_section(el):
        return []
    classes = set(el.get("class", []))

    if el.name == "figure":
        blocks = _figure_blocks(el)
    elif el.name == "table" and classes & {"ltx_equation", "ltx_equationgroup"}:
        blocks = [_block("equation", _equation_text(el))]
    elif "ltx_tabular" in classes:
        blocks = [_block("table", _table_text(el))]
    elif el.name in {"ul", "ol", "dl"}:
        blocks = [_block("list", _list_text(el))]
    elif el.name == "pre" or classes & {"ltx_listing", "ltx_verbatim"}:
        blocks = [_block("code", _code_text(el))]
    elif el.name == "p" or not _has_block_children(el):
        text = LEAKED_FOOTNOTETEXT.sub(r"[Footnote: \1]", _inline_text(el))
        blocks = [_block("paragraph", text)]
    else:  # a container (div.ltx_para, theorem, blockquote, ...): look inside
        blocks = []
        for child in el.children:
            if isinstance(child, Tag):
                blocks.extend(_blocks_from(child))
            elif child.strip():
                blocks.append(_block("paragraph", normalize_text(str(child))))

    return [b for b in blocks if b["text"]]


def _has_block_children(el: Tag) -> bool:
    return any(isinstance(c, Tag) and c.name in BLOCK_TAGS for c in el.children)


def _inline_text(el: Tag) -> str:
    """Text of an element, with its footnotes moved to the end."""
    footnotes = []
    for note in el.find_all("span", class_="ltx_note"):
        content = note.find(class_="ltx_note_content")
        if content is not None:
            for marker in content.find_all(class_=["ltx_note_mark", "ltx_tag_note"]):
                marker.decompose()
            footnote = normalize_text(content.get_text(""))
            if footnote:
                footnotes.append(f"[Footnote: {footnote}]")
        note.decompose()
    return normalize_text(" ".join([el.get_text(""), *footnotes]))


def _figure_blocks(fig: Tag) -> list[dict]:
    # A figure can hold several sub-figures/sub-tables, each with its own caption (e.g. a whole
    # appendix of example tables). Emit one block per sub-figure instead of one giant block.
    panels = [f for f in fig.find_all("figure") if f.find_parent("figure") is fig]
    if panels:
        blocks = []
        own_captions = [_inline_text(c) for c in fig.find_all("figcaption", recursive=False)]
        own_tables = [t for t in fig.select(".ltx_tabular") if t.find_parent("figure") is fig]
        if own_tables:  # the outer figure also has its own table next to the panels
            rows = [
                _table_text(t) for t in own_tables if t.find_parent(class_="ltx_tabular") is None
            ]
            blocks.append(_block("table", "\n".join([*own_captions, *rows])))
        elif any(own_captions):
            blocks.append(_block("figure", " ".join(c for c in own_captions if c)))
        for panel in panels:
            blocks.extend(_figure_blocks(panel))
        return blocks

    if fig.select(".ltx_tabular"):
        return [_block("table", _table_text(fig))]
    blocks = []
    listing = fig.select_one(".ltx_listing, pre")
    if listing is not None:
        blocks.append(_block("code", _code_text(listing)))
    captions = [_inline_text(c) for c in fig.find_all("figcaption")]
    if captions:
        blocks.append(_block("figure", " ".join(c for c in captions if c)))
    return blocks


def _table_text(el: Tag) -> str:
    """Caption first, then one line per row with cells separated by ' | '."""
    lines = [_inline_text(c) for c in el.find_all("figcaption")]
    for row in el.select(".ltx_tr"):
        cells = row.select(":scope > .ltx_td, :scope > td, :scope > th")
        texts = [_inline_text(cell) for cell in cells]
        if any(texts):
            lines.append(" | ".join(texts))
    return "\n".join(line for line in lines if line)


def _equation_text(table: Tag) -> str:
    lines = []
    for row in table.find_all("tr"):
        parts = [normalize_text(td.get_text("")) for td in row.find_all("td")]
        line = " ".join(p for p in parts if p)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _list_text(el: Tag) -> str:
    items = [_inline_text(item) for item in el.find_all(["li", "dt", "dd"], recursive=False)]
    return "\n".join(item for item in items if item)


def _code_text(el: Tag) -> str:
    lines = el.select(".ltx_listingline")
    if lines:
        return "\n".join(line.get_text("").rstrip() for line in lines).strip()
    return el.get_text("").strip()

"""Turn the generator's [n] markers into sources: paper title + section + link.

The answer keeps the model's own numbers, so "[5]" in the text is source [5] in the list
(chunk n is chunks[n - 1]). Only chunks the answer actually cites are listed, in the order
they are first cited. Each source links straight to its section in the pinned arXiv HTML
version (the abstract page for abstract chunks).
"""

from dataclasses import dataclass, field

from ragpapers.generation.prompt import is_abstention, parse_citations


@dataclass(frozen=True)
class Source:
    number: int  # the [n] used in the answer
    chunk_id: str
    paper_id: str
    title: str
    year: int
    section: str
    url: str


@dataclass(frozen=True)
class CitedAnswer:
    text: str  # the generator's reply, unchanged
    abstained: bool
    sources: list[Source] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)  # numbers never shown


def source_for(number: int, chunk: dict) -> Source:
    return Source(
        number=number,
        chunk_id=chunk["chunk_id"],
        paper_id=chunk["paper_id"],
        title=chunk["title"],
        year=chunk["year"],
        section=chunk["section"],
        url=chunk["section_url"],
    )


def cite_answer(reply: str, chunks: list[dict], abstention_text: str) -> CitedAnswer:
    """The reply with its cited sources; an abstention has none."""
    text = reply.strip()
    if is_abstention(text, abstention_text):
        return CitedAnswer(text=abstention_text, abstained=True)
    citations = parse_citations(text, len(chunks))
    return CitedAnswer(
        text=text,
        abstained=False,
        sources=[source_for(n, chunks[n - 1]) for n in citations.valid],
        invalid_citations=citations.invalid,
    )


def format_source(source: Source) -> str:
    return f"[{source.number}] {source.title} ({source.year}), {source.section}. {source.url}"


def format_sources(sources: list[Source]) -> str:
    """One line per source, in the order given (first citation first)."""
    return "\n".join(format_source(s) for s in sources)

"""Display helpers of the Streamlit page: pure functions, so they are tested without a browser.

Display only: nothing here changes what the API returned. Citation markers are normalised
and drawn as badges on screen; the stored answer (API response, query log) is unchanged.
"""

import html
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ragpapers.config import load_config
from ragpapers.generation.prompt import CITATION, is_abstention
from ragpapers.ingest.manifest import load_manifest
from ragpapers.ui import latex

ABSTENTION_TEXT = load_config()["abstention_text"]

# Example chips: answerable DEV questions only (never the held-out test split), worded exactly
# as in eval/questions.jsonl, so the answers come from the LLM cache, and only those whose
# recorded hybrid dev answer did not abstain. tests/test_display.py checks all of it.
EXAMPLES = [
    (
        "REALM index refresh",
        "How often does REALM rebuild its document index while the retriever is being trained?",
    ),
    (
        "monoBERT query length",
        "When monoBERT scores a query against a passage, how long can the query be "
        "before it gets cut off?",
    ),
    (
        "C4 cleaning",
        "When cleaning Common Crawl into C4, what did the T5 authors do with pages "
        "containing placeholder text or source code?",
    ),
    (
        "ELECTRA weight sharing",
        "ELECTRA's generator is smaller than its discriminator. "
        "Which weights do the two networks still share?",
    ),
    (
        "BERT sequence lengths",
        "To save compute, which sequence lengths did BERT pre-train with, "
        "and for what share of the steps?",
    ),
    ("SimCSE weakness on MTEB", "Which weakness of SimCSE did the MTEB benchmark reveal?"),
]


@dataclass(frozen=True)
class TopicStyle:
    name: str  # display name
    dot: str  # sidebar dot colour
    tint: str  # topic pill background (light)
    ink: str  # topic pill text (dark; >= 7:1 on the tint)


# Manifest topic tag -> how the topic is shown. Unknown tags fall back to OTHER_TOPIC.
TOPIC_STYLES = {
    "rag-retrieval": TopicStyle("RAG & retrieval", "#38bdf8", "#e0f2fe", "#075985"),
    "instruction-rlhf": TopicStyle("Instruction tuning & RLHF", "#f97316", "#ffedd5", "#9a3412"),
    "pretraining": TopicStyle("Pre-training", "#a78bfa", "#ede9fe", "#5b21b6"),
    "embeddings": TopicStyle("Sentence embeddings", "#34d399", "#d1fae5", "#065f46"),
    "reasoning-agents": TopicStyle("Reasoning & agents", "#f472b6", "#fce7f3", "#9d174d"),
    "transformers": TopicStyle("Transformers & attention", "#fbbf24", "#fef3c7", "#92400e"),
    "peft": TopicStyle("Parameter-efficient fine-tuning", "#2dd4bf", "#ccfbf1", "#115e59"),
    "scaling": TopicStyle("Scaling laws", "#fb7185", "#ffe4e6", "#9f1239"),
    "open-models": TopicStyle("Open-weight models", "#818cf8", "#e0e7ff", "#3730a3"),
}
OTHER_TOPIC = TopicStyle("Other", "#9ca3af", "#f3f4f6", "#374151")

# Citation badge colours, cycling in citation order (white text: 5.5:1 or better on each).
BADGE_COLORS = ["#4338ca", "#0f766e", "#6d28d9"]

_PLAIN_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Corpus:
    n_papers: int
    first_year: int
    last_year: int
    topics: list[tuple[str, int]]  # (manifest tag, paper count), most papers first
    topic_of: dict[str, str]  # arXiv ID (unversioned) -> manifest tag


def load_corpus(manifest_path: str | Path) -> Corpus | None:
    """Paper count, year range and topics from the manifest; None if the file is missing."""
    path = Path(manifest_path)
    if not path.exists():
        return None
    papers = load_manifest(path)
    counts = Counter(p.topic for p in papers)
    topics = sorted(counts.items(), key=lambda item: (-item[1], topic_style(item[0]).name))
    return Corpus(
        n_papers=len(papers),
        first_year=min(p.year for p in papers),
        last_year=max(p.year for p in papers),
        topics=topics,
        topic_of={p.arxiv_id: p.topic for p in papers},
    )


def topic_style(tag: str) -> TopicStyle:
    return TOPIC_STYLES.get(tag, OTHER_TOPIC)


def normalize_citations(text: str) -> str:
    """Every citation form the frozen parser reads, written as "[n]" or "[n, m]".

    "【3】" -> "[3]", "【3, 5】" -> "[3, 5]", "【3†L1-L4】" -> "[3]" (the part after † is not
    a passage number). Uses the parser's own pattern, so display and scoring agree.
    """

    def plain(match: re.Match) -> str:
        numbers = [part.strip() for part in match.group(1).split(",")]
        return "[" + ", ".join(numbers) + "]"

    return CITATION.sub(plain, text)


def badge_colors(cited: list[int]) -> dict[int, str]:
    """Passage number -> badge colour, cycling through BADGE_COLORS in citation order.

    `cited` is the API's citation order (first-cited first), so a passage keeps the same
    colour in the answer and on its source card.
    """
    return {n: BADGE_COLORS[i % len(BADGE_COLORS)] for i, n in enumerate(cited)}


def answer_html(text: str, colors: dict[int, str]) -> str:
    """The answer as Markdown + HTML: citation badges, and math rendered by KaTeX.

    Prose is HTML-escaped (it comes from a model), keeps its Markdown (bold, lists), and each
    valid citation (a number in `colors`) becomes a coloured badge; any other "[n]" stays
    plain text. Formulas ($...$, $$...$$, \\(...\\), \\[...\\]) are rendered; one KaTeX
    could not render is shown as plain text (inline code).
    """

    def badges(match: re.Match) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        if not all(n in colors for n in numbers):
            return match.group(0)
        return "".join(
            f'<span class="cite-badge" style="background:{colors[n]}">{n}</span>' for n in numbers
        )

    out = []
    for piece in latex.split_math(text, brackets=True):
        if piece.kind == "text":
            escaped = html.escape(normalize_citations(piece.text), quote=False)
            # a lone "$" (e.g. an amount) must not start math
            out.append(_PLAIN_CITATION.sub(badges, escaped).replace("$", r"\$"))
        elif piece.kind == "raw":
            out.append(f"<code>{html.escape(piece.text)}</code>")
        else:
            out.append(latex.math_markdown(piece))
    return "".join(out)


def is_abstention_response(response: dict) -> bool:
    """True if the API says the system abstained, or the answer is the abstention sentence."""
    return bool(response.get("abstained")) or is_abstention(
        response.get("answer", ""), ABSTENTION_TEXT
    )


def section_path(section: str) -> str:
    """The section path with "›" separators: "3 Experiments › 3.2 Model Extensions"."""
    return " › ".join(part.strip() for part in section.split(">"))


def seconds(ms: float) -> str:
    """A duration for the footer: "< 0.01 s", "0.04 s", "1.8 s", "12 s"."""
    s = ms / 1000
    if s < 0.01:
        return "< 0.01 s"
    if s < 1:
        return f"{s:.2f} s"
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


def sources_cited(n: int) -> str:
    return f"{n} source cited" if n == 1 else f"{n} sources cited"


def footer_html(response: dict) -> str:
    """The card footer: generation time, model and passages retrieved, plus a "cached" pill.

    "Generated in" is `latency_ms.generation`, the API's measurement of the generator call
    alone (~0 when the answer comes from the cache), not the request total: the total also
    includes retrieval, whose first query after the API starts loads the embedding model.
    """
    n = len(response["retrieved"])
    text = (
        f"Generated in {seconds(response['latency_ms']['generation'])} · {response['model']} · "
        f"{n} passage{'s' if n != 1 else ''} retrieved"
    )
    pill = '<span class="cached-pill">cached</span>' if response["cached"] else ""
    return f'<div class="card-footer">{html.escape(text)}{pill}</div>'


# --- HTML pieces of the page (all text HTML-escaped) ---------------------------------------------

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'aria-hidden="true">{}</svg>'
)
ICONS = {  # Lucide icons (ISC licence), drawn in the current text colour
    "book": _SVG.format(
        '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
        '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'
    ),
    "check": _SVG.format(
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>'
    ),
    "search": _SVG.format('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'),
    "external": _SVG.format(
        '<path d="M15 3h6v6"/><path d="M10 14 21 3"/>'
        '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
    ),
}


def sidebar_html(corpus: Corpus | None) -> str:
    """Brand row, corpus card and topic list (topics sorted by paper count)."""
    parts = [
        '<div class="brand-row">'
        f'<div class="brand-icon">{ICONS["book"]}</div>'
        '<div><div class="brand-name">PaperRAG</div>'
        '<div class="brand-sub">Research paper assistant</div></div></div>'
    ]
    if corpus is None:
        return "".join(parts)
    parts.append(
        '<div class="corpus-card"><div class="eyebrow">Corpus</div>'
        f'<div class="corpus-count">{corpus.n_papers} papers</div>'
        f'<div class="corpus-sub">AI/ML research from arXiv, '
        f"{corpus.first_year}–{corpus.last_year}</div></div>"
        '<div class="eyebrow topics-label">Topics</div>'
    )
    for tag, count in corpus.topics:
        style = topic_style(tag)
        parts.append(
            f'<div class="topic-row"><span class="topic-dot" style="background:{style.dot}">'
            f'</span><span class="topic-name">{html.escape(style.name)}</span>'
            f'<span class="topic-count">{count}</span></div>'
        )
    return "".join(parts)


def hero_html(corpus: Corpus | None) -> str:
    papers = f"{corpus.n_papers} AI/ML papers" if corpus else "AI/ML research papers"
    return (
        f'<span class="hero-pill">Grounded in {papers}</span>'
        '<div class="hero-title">AI/ML Research Assistant</div>'
        '<div class="hero-sub">Every answer cites the exact passages it relies on. When the '
        "papers don't cover a question, the assistant says so instead of guessing.</div>"
    )


def answer_header_html(n_sources: int) -> str:
    return (
        '<div class="card-header answer-header">'
        f'<span class="card-title">{ICONS["check"]}Answer</span>'
        f'<span class="card-meta">{sources_cited(n_sources)}</span></div>'
    )


def abstention_html(response: dict) -> str:
    """Header bar and body of the "not found" card (the footer is added separately)."""
    n = len(response["retrieved"])
    return (
        '<div class="card-header abstain-header">'
        f'<span class="card-title">{ICONS["search"]}Not found in the papers</span></div>'
        '<div class="card-body">'
        f'<p class="abstain-main">{html.escape(ABSTENTION_TEXT)}</p>'
        f'<p class="abstain-hint">The {n} most relevant passages were checked and none answers '
        "this question. Try rephrasing, or ask about one of the topics on the left.</p></div>"
    )


def source_head_html(source: dict, color: str, corpus: Corpus | None) -> str:
    """Number badge, title (year), section path, topic pill and the "Open on arXiv" link."""
    tag = corpus.topic_of.get(source["paper_id"]) if corpus else None
    pill = ""
    if tag is not None:
        style = topic_style(tag)
        pill = (
            f'<span class="topic-pill" style="background:{style.tint};color:{style.ink}">'
            f"{html.escape(style.name)}</span>"
        )
    return (
        '<div class="src-row">'
        f'<span class="num-badge" style="background:{color}">{source["number"]}</span>'
        '<div class="src-main">'
        f'<div class="src-title">{html.escape(source["title"])} '
        f'<span class="src-year">({source["year"]})</span></div>'
        f'<div class="src-section">{html.escape(section_path(source["section"]))}</div>'
        f"{pill}</div>"
        f'<a class="arxiv-btn" href="{html.escape(source["url"], quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{ICONS["external"]}Open on arXiv</a>'
        '</div><div class="card-divider"></div>'
    )

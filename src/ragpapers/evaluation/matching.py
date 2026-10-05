"""Decide whether a piece of text contains a gold evidence quote.

Used twice:
- to check that every evidence quote in the eval set really exists in the corpus.
- to decide whether a retrieved chunk is relevant (it contains an evidence quote).

Both sides are normalized the same way, then compared with rapidfuzz's partial_ratio: the score
(0-100) of the best-matching window of the longer text against the quote.
"""

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from ragpapers.corpus import iter_blocks

QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
DASH_CHARS = str.maketrans({"–": "-", "—": "-", "−": "-"})
NUMERIC_CITATION_INLINE = re.compile(r"\[\s*\d+(\s*[,-]\s*\d+)*\s*\]")


def normalize_for_matching(text: str) -> str:
    """Make a quote and a corpus text comparable, whichever page the quote was copied from.

    Lowercase; straighten quotes and dashes; drop citation markers, both our "[ref]" placeholder
    and numeric ones like "[12]" copied from the rendered arXiv page; drop the $ math delimiters
    (so "$d$-dimensional" matches "d-dimensional"); collapse whitespace, including the space a
    removed marker leaves before punctuation.
    """
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_CHARS).translate(DASH_CHARS)
    text = text.lower().replace("[ref]", " ").replace("$", "")
    text = NUMERIC_CITATION_INLINE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r" ([,.;:)\]])", r"\1", text).strip()


def _score_normalized(q: str, t: str) -> float:
    if not q or not t:
        return 0.0
    if q in t:
        return 100.0
    if len(t) < len(q):
        # partial_ratio would slide the *shorter* string, so a block like "and" would score 100
        # against any quote containing "and". A text shorter than the quote cannot contain it:
        # compare whole strings instead.
        return fuzz.ratio(q, t)
    return fuzz.partial_ratio(q, t)


def quote_score(quote: str, text: str) -> float:
    """How well `text` contains `quote` (0-100). 100 = verbatim after normalization."""
    return _score_normalized(normalize_for_matching(quote), normalize_for_matching(text))


def contains_quote(quote: str, text: str, threshold: float) -> bool:
    return quote_score(quote, text) >= threshold


@dataclass
class QuoteMatch:
    score: float
    exact: bool  # the normalized quote is a verbatim substring of one block
    section_id: str
    section_path: list[str]
    block_text: str


def locate_quote(quote: str, paper: dict) -> QuoteMatch | None:
    """Best-matching block of a parsed paper for a quote (None if the paper has no blocks)."""
    q = normalize_for_matching(quote)
    best = None
    for section, block in iter_blocks(paper):
        t = normalize_for_matching(block["text"])
        exact = bool(q) and q in t
        score = _score_normalized(q, t)
        if best is None or score > best.score:
            best = QuoteMatch(score, exact, section["section_id"], section["path"], block["text"])
            if exact:
                break
    return best

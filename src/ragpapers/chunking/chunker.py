"""Section-aware chunking with overlap.

Input: a parsed paper from data/processed/<paper_id>.json (sections of typed blocks).
Output: chunks that never cross a section boundary, so every chunk has one section to cite.

Rules, applied per section:
1. Blocks are kept whole when possible: a paragraph, table or equation is never cut unless it is
   longer than a chunk can hold.
2. A block that is too long is split with `split_block`: tables, code and lists on line
   boundaries, other text on sentence boundaries, and as a last resort into fixed windows of
   whole words. A single word longer than a window (only giant LaTeX expressions written
   without spaces) is cut into character slices with `split_long_word`.
3. Pieces are packed greedily into chunks of at most `chunk_size` tokens, joined with "\\n".
4. Each new chunk of a section starts with the longest run of whole words from the end of the
   previous chunk that fits in `overlap` tokens (`overlap_tail`). A word is never split.
"""

import re
from collections.abc import Callable

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")  # split after . ! ? followed by whitespace
LINE_TYPES = {"table", "code", "list"}  # block types split on line boundaries, not sentences

CountTokens = Callable[[str], int]


def word_count(text: str) -> int:
    return len(text.split())


def split_long_word(word: str, max_tokens: int, count_tokens: CountTokens) -> list[str]:
    """Cut one word that is longer than `max_tokens` into character slices that each fit.

    This is the only place where a word is ever split, and it only happens for "words" that
    are themselves longer than a whole window.

    Example (a fake counter where a word costs ceil(len / 3) tokens), max_tokens=2:
        "abcdefghij" (4 tokens) -> ["abcdef", "ghij"]
    """
    if count_tokens(word) <= max_tokens:
        return [word]

    slices: list[str] = []
    start = 0
    while start < len(word):
        end = start + 1  # always take at least one character, so the loop always advances
        while end < len(word) and count_tokens(word[start : end + 1]) <= max_tokens:
            end += 1
        slices.append(word[start:end])
        start = end
    return slices


def split_block(
    text: str, block_type: str, max_tokens: int, count_tokens: CountTokens
) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    # Cut into natural units: lines for tables/code/lists, sentences for everything else.
    if block_type in LINE_TYPES:
        units = text.split("\n")
        joiner = "\n"
    else:
        units = SENTENCE_END.split(text)
        joiner = " "
    units = [u for u in units if u.strip()]  # drop empty lines

    # A unit that is still too long is cut into windows of whole words of at most max_tokens.
    # A word that alone exceeds max_tokens is cut into character slices (giant LaTeX only).
    small_units: list[str] = []
    for unit in units:
        if count_tokens(unit) <= max_tokens:
            small_units.append(unit)
            continue
        window: list[str] = []
        window_tokens = 0
        for word in unit.split():
            n = count_tokens(word)
            if n > max_tokens:
                if window:
                    small_units.append(" ".join(window))
                    window, window_tokens = [], 0
                small_units.extend(split_long_word(word, max_tokens, count_tokens))
                continue
            if window and window_tokens + n > max_tokens:
                small_units.append(" ".join(window))
                window, window_tokens = [], 0
            window.append(word)
            window_tokens += n
        if window:
            small_units.append(" ".join(window))

    # Pack whole units greedily, in order, while the piece stays <= max_tokens.
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in small_units:
        n = count_tokens(unit)
        if current and current_tokens + n > max_tokens:
            pieces.append(joiner.join(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += n
    if current:
        pieces.append(joiner.join(current))

    return pieces


def overlap_tail(text: str, overlap: int, count_tokens: CountTokens) -> str:
    tail: list[str] = []
    tail_tokens = 0
    for word in reversed(text.split()):
        n = count_tokens(word)
        if tail_tokens + n > overlap:
            break
        tail.append(word)
        tail_tokens += n
    return " ".join(reversed(tail))


def chunk_section(
    blocks: list[dict], chunk_size: int, overlap: int, count_tokens: CountTokens
) -> list[str]:
    if not 0 <= overlap < chunk_size:
        raise ValueError(
            f"need 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}"
        )

    # Split oversized blocks so that "overlap prefix + any piece" always fits.
    max_piece = chunk_size - overlap
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(split_block(block["text"], block["type"], max_piece, count_tokens))

    # Pack pieces greedily; each new chunk starts with the previous chunk's last words.
    chunks: list[str] = []
    current = ""
    current_tokens = 0
    for piece in pieces:
        n = count_tokens(piece)
        if n == 0:
            continue  # skip empty blocks
        if current_tokens == 0:
            current, current_tokens = piece, n  # first piece of the section: no prefix
        elif current_tokens + n <= chunk_size:
            current += "\n" + piece
            current_tokens += n
        else:
            chunks.append(current)
            tail = overlap_tail(current, overlap, count_tokens) if overlap > 0 else ""
            if tail:
                current = tail + "\n" + piece
                current_tokens = count_tokens(tail) + n
            else:
                current, current_tokens = piece, n
    if current_tokens > 0:
        chunks.append(current)

    return chunks


def chunk_paper(
    paper: dict,
    chunk_size: int,
    overlap: int,
    count_tokens: CountTokens,
    include_appendix: bool = True,
) -> list[dict]:
    """Chunk every section of a parsed paper, adding the metadata each chunk needs.

    Sections are processed in document order with `chunk_section`. Appendix sections
    (`section["is_appendix"]`) are skipped when include_appendix is False.

    Each chunk is a dict with exactly these keys:
        chunk_id     f"{paper_id}:{n:04d}", n counting chunks across the whole paper from 0
        paper_id     paper["paper_id"]
        title        paper["title"]
        year         paper["year"]
        section      the section path joined with " > ", e.g. "3 Model Architecture > 3.2 Attention"
        section_url  section["url"] (links straight to the section)
        source_url   paper["source_url"] (the arXiv abstract page)
        is_appendix  section["is_appendix"]
        text         the chunk text
        n_words      word_count(text)
        n_tokens     count_tokens(text), the chunk body only (no header, no special tokens)
    """
    chunks: list[dict] = []
    n = 0  # chunk counter across the whole paper

    for section in paper["sections"]:
        if section["is_appendix"] and not include_appendix:
            continue
        for text in chunk_section(section["blocks"], chunk_size, overlap, count_tokens):
            chunks.append(
                {
                    "chunk_id": f"{paper['paper_id']}:{n:04d}",
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "year": paper["year"],
                    "section": " > ".join(section["path"]),
                    "section_url": section["url"],
                    "source_url": paper["source_url"],
                    "is_appendix": section["is_appendix"],
                    "text": text,
                    "n_words": word_count(text),
                    "n_tokens": count_tokens(text),
                }
            )
            n += 1

    return chunks

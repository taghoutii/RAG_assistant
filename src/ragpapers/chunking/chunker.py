"""Section-aware chunking with overlap.

Input: a parsed paper from data/processed/<paper_id>.json (sections of typed blocks).
Output: chunks that never cross a section boundary, so every chunk has one section to cite.

All lengths are in WORDS (`len(text.split())`): `chunk_size` and `overlap` are word counts.
The embedding model's token limit is checked separately when the index is built.

Rules, applied per section:
1. Blocks are kept whole when possible: a paragraph, table or equation is never cut unless it is
   longer than a chunk can hold.
2. A block that is too long is split with `split_block`: tables, code and lists on line
   boundaries, other text on sentence boundaries, and as a last resort into fixed word windows.
3. Pieces are packed greedily into chunks of at most `chunk_size` words, joined with "\\n".
4. Each new chunk of a section starts with the last `overlap` words of the previous chunk.
"""

import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")  # split after . ! ? followed by whitespace
LINE_TYPES = {"table", "code", "list"}  # block types split on line boundaries, not sentences


def word_count(text: str) -> int:
    return len(text.split())


def split_block(text: str, block_type: str, max_words: int) -> list[str]:
    """Split one block into pieces of at most `max_words` words each.

    - If the block already fits, return [text] unchanged.
    - Tables, code and lists (`LINE_TYPES`): split on "\\n" into lines (a list has one item per
      line), then pack whole lines greedily, in order, joining them with "\\n", while the piece
      stays <= max_words.
    - Any other type: split into sentences with `SENTENCE_END`, then pack whole sentences
      greedily, joining them with " ", while the piece stays <= max_words.
    - A single line or sentence longer than max_words is first cut into consecutive windows of
      max_words words (joined with " "); each window is then packed like any other unit.

    Examples:
        max_words=6, paragraph: "One two three. Four five six. Seven eight."
            -> ["One two three. Four five six.", "Seven eight."]
        max_words=4, table: "a b\\nc d\\ne f"
            -> ["a b\\nc d", "e f"]
        max_words=7, list: "• a b c\\n• d e\\n• f g h"
            -> ["• a b c\\n• d e", "• f g h"]
    """
    if word_count(text) <= max_words:
        return [text]

    # Cut into natural units: lines for tables/code, sentences for everything else.
    if block_type in LINE_TYPES:
        units = text.split("\n")
        joiner = "\n"
    else:
        units = SENTENCE_END.split(text)
        joiner = " "
    units = [u for u in units if u.strip()]  # drop empty lines

    # A unit that is still too long is cut into fixed windows of max_words words.
    small_units: list[str] = []
    for unit in units:
        if word_count(unit) <= max_words:
            small_units.append(unit)
        else:
            words = unit.split()
            for start in range(0, len(words), max_words):
                small_units.append(" ".join(words[start : start + max_words]))

    # Pack whole units greedily, in order, while the piece stays <= max_words.
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in small_units:
        n = word_count(unit)
        if current and current_words + n > max_words:
            pieces.append(joiner.join(current))
            current = []
            current_words = 0
        current.append(unit)
        current_words += n
    if current:
        pieces.append(joiner.join(current))

    return pieces


def chunk_section(blocks: list[dict], chunk_size: int, overlap: int) -> list[str]:
    """Pack the blocks of ONE section into chunk texts.

    Steps:
    1. Every block longer than (chunk_size - overlap) words is split with
       split_block(text, type, max_words=chunk_size - overlap). This guarantees that
       "overlap prefix + any piece" always fits in chunk_size.
    2. Pieces are added in order to the current chunk, joined with "\\n", while the chunk stays
       <= chunk_size words.
    3. When the next piece does not fit, emit the current chunk. The next chunk starts with the
       last `overlap` words of the emitted chunk (joined with " "; fewer if the chunk is shorter),
       then "\\n", then the piece. With overlap == 0 there is no prefix.
    4. The first chunk of a section has no prefix. Emit the last chunk at the end.

    Returns [] for a section without blocks.
    Raises ValueError unless 0 <= overlap < chunk_size.

    Example (chunk_size=6, overlap=2), blocks "a1 a2 a3", "b1 b2", "c1 c2 c3":
        -> ["a1 a2 a3\\nb1 b2", "b1 b2\\nc1 c2 c3"]
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError(
            f"need 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}"
        )

    # Split oversized blocks so that "overlap prefix + any piece" always fits.
    max_piece = chunk_size - overlap
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(split_block(block["text"], block["type"], max_piece))

    # Pack pieces greedily; each new chunk starts with the previous chunk's last words.
    chunks: list[str] = []
    current = ""
    current_words = 0
    for piece in pieces:
        n = word_count(piece)
        if n == 0:
            continue  # skip empty blocks
        if current_words == 0:
            current, current_words = piece, n  # first piece of the section: no prefix
        elif current_words + n <= chunk_size:
            current += "\n" + piece
            current_words += n
        else:
            chunks.append(current)
            tail = current.split()[-overlap:] if overlap > 0 else []
            if tail:
                current = " ".join(tail) + "\n" + piece
                current_words = len(tail) + n
            else:
                current, current_words = piece, n
    if current_words > 0:
        chunks.append(current)

    return chunks


def chunk_paper(
    paper: dict, chunk_size: int, overlap: int, include_appendix: bool = True
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
    """
    chunks: list[dict] = []
    n = 0  # chunk counter across the whole paper

    for section in paper["sections"]:
        if section["is_appendix"] and not include_appendix:
            continue
        for text in chunk_section(section["blocks"], chunk_size, overlap):
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
                }
            )
            n += 1

    return chunks

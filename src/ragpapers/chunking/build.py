"""Write the chunk files used by every index: data/chunks/<config>.jsonl.

Two texts are indexed per chunk; the chunk `text` itself is what quote matching, citations and
display use:
- BM25 sees `bm25_text`: the FULL header "<title> > <full section path>" + the chunk. BM25 has
  no length limit, and the extra words help keyword matching.
- The embedding model sees `dense_text`: a SHORT header "<short title> > <section heading>",
  capped at `dense_header.max_tokens`, + the chunk. It must fit the model's 512-token window.

The build checks the hard guarantee: header + chunk + special tokens <= max_sequence_tokens for
every chunk (it fails otherwise), and reports how many eval quotes no single chunk contains.
"""

import json
import sys
from pathlib import Path

from ragpapers.chunking.chunker import chunk_paper, word_count
from ragpapers.chunking.tokens import TokenCounter
from ragpapers.config import REPO_ROOT, load_config
from ragpapers.corpus import load_corpus
from ragpapers.evaluation.matching import contains_quote
from ragpapers.ingest.manifest import load_manifest

UNITS = ("tokens", "words")
ELLIPSIS = " …"


def config_name(unit: str, chunk_size: int, overlap: int) -> str:
    """'tokens256_ov64' for token sizing; 'size150_ov30' (the Phase 3 name) for word sizing."""
    prefix = "tokens" if unit == "tokens" else "size"
    return f"{prefix}{chunk_size}_ov{overlap}"


# --- index texts -------------------------------------------------------------------------------


def short_title(title: str) -> str:
    """The part before the first ":" ("LoRA: Low-Rank ..." -> "LoRA"); whole title otherwise."""
    return title.split(":")[0].strip()


def make_dense_header(
    title: str, section: str, header_format: str, max_tokens: int, count_tokens
) -> str:
    """Short header for the embedding model, at most `max_tokens` tokens.

    `header_format` fields: {short_title}, {title}, {section} (the last heading of the path) and
    {path} (the full path). A header over the cap loses words from the end and gets " …".
    """
    header = header_format.format(
        short_title=short_title(title),
        title=title,
        section=section.split(" > ")[-1],
        path=section,
    )
    if count_tokens(header) <= max_tokens:
        return header
    words = header.split()
    while words and count_tokens(" ".join(words) + ELLIPSIS) > max_tokens:
        words.pop()
    return " ".join(words) + ELLIPSIS


def bm25_text(chunk: dict) -> str:
    """What BM25 indexes: full title + full section path + chunk text."""
    return f"{chunk['title']} > {chunk['section']}\n{chunk['text']}"


def dense_text(chunk: dict) -> str:
    """What the embedding model sees: the capped short header + chunk text."""
    return f"{chunk['dense_header']}\n{chunk['text']}"


# --- building and checking ---------------------------------------------------------------------


def build_chunks(
    corpus: dict[str, dict], chunk_size: int, overlap: int, count_tokens, header_cfg: dict
) -> list[dict]:
    """Chunk every paper (appendices included), in manifest order, adding `dense_header`."""
    chunks = []
    for paper in corpus.values():
        for chunk in chunk_paper(paper, chunk_size, overlap, count_tokens, include_appendix=True):
            chunk["dense_header"] = make_dense_header(
                chunk["title"],
                chunk["section"],
                header_cfg["format"],
                header_cfg["max_tokens"],
                count_tokens,
            )
            chunks.append(chunk)
    return chunks


def over_budget(
    chunks: list[dict], count_tokens, special_tokens: int, max_sequence_tokens: int
) -> list[tuple[str, int]]:
    """(chunk_id, sequence length) of every chunk whose embedded sequence is too long.

    The full string is re-counted exactly: dense header + "\\n" + text + special tokens.
    """
    over = []
    for chunk in chunks:
        n = count_tokens(dense_text(chunk)) + special_tokens
        if n > max_sequence_tokens:
            over.append((chunk["chunk_id"], n))
    return over


def quotes_cut(chunks: list[dict], questions: list[dict], threshold: float) -> list[str]:
    """'<question id>/<paper id>' for every evidence quote that no single chunk contains."""
    by_paper: dict[str, list[str]] = {}
    for chunk in chunks:
        by_paper.setdefault(chunk["paper_id"], []).append(chunk["text"])
    cut = []
    for q in questions:
        for ev in q["evidence"]:
            texts = by_paper.get(ev["paper_id"], [])
            if not any(contains_quote(ev["quote"], t, threshold) for t in texts):
                cut.append(f"{q['id']}/{ev['paper_id']}")
    return cut


def validate_chunking_config(config: dict, special_tokens: int) -> None:
    """Fail early on settings that cannot work. Raises ValueError with the reason."""
    chunking = config["chunking"]
    if chunking["unit"] not in UNITS:
        raise ValueError(f"chunking.unit must be one of {UNITS}, got {chunking['unit']!r}")
    if chunking["tokenizer"] != config["embedding"]["model"]:
        raise ValueError(
            f"chunking.tokenizer ({chunking['tokenizer']}) must be the embedding model "
            f"({config['embedding']['model']}), or token counts would not match what it reads"
        )
    budget = (
        chunking["max_sequence_tokens"] - special_tokens - chunking["dense_header"]["max_tokens"]
    )
    for setting in chunking["compare"]:
        size, overlap = setting["chunk_size"], setting["overlap"]
        if not 0 <= overlap < size:
            raise ValueError(f"need 0 <= overlap < chunk_size, got {setting}")
        if chunking["unit"] == "tokens" and size > budget:
            raise ValueError(
                f"chunk_size {size} exceeds the body budget of {budget} tokens "
                f"({chunking['max_sequence_tokens']} - {special_tokens} special - "
                f"{chunking['dense_header']['max_tokens']} header)"
            )


def write_chunks(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(c, ensure_ascii=False) for c in chunks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunks_path(config: dict, unit: str, chunk_size: int, overlap: int) -> Path:
    name = config_name(unit, chunk_size, overlap)
    return REPO_ROOT / config["paths"]["chunks_dir"] / f"{name}.jsonl"


def main() -> None:
    from ragpapers.evaluation.questions import load_questions

    config = load_config()
    chunking = config["chunking"]
    counter = TokenCounter(chunking["tokenizer"])
    validate_chunking_config(config, counter.special_tokens)
    count_tokens = counter if chunking["unit"] == "tokens" else word_count

    paper_ids = [p.arxiv_id for p in load_manifest(REPO_ROOT / config["paths"]["manifest"])]
    corpus = load_corpus(REPO_ROOT / config["paths"]["processed_dir"], paper_ids)
    questions = load_questions(REPO_ROOT / config["paths"]["eval_questions"])
    n_quotes = sum(len(q["evidence"]) for q in questions)
    threshold = config["retrieval_eval"]["quote_match_threshold"]

    failed = False
    for setting in chunking["compare"]:
        size, overlap = setting["chunk_size"], setting["overlap"]
        chunks = build_chunks(corpus, size, overlap, count_tokens, chunking["dense_header"])
        path = chunks_path(config, chunking["unit"], size, overlap)
        write_chunks(path, chunks)

        over = over_budget(chunks, counter, counter.special_tokens, chunking["max_sequence_tokens"])
        cut = quotes_cut(chunks, questions, threshold)
        longest = max(counter(dense_text(c)) + counter.special_tokens for c in chunks)
        print(
            f"{path.name}: {len(chunks)} chunks ({sum(c['is_appendix'] for c in chunks)} from "
            f"appendices); longest embedded sequence {longest} tokens"
        )
        print(f"  {len(over)} chunks over the {chunking['max_sequence_tokens']}-token limit")
        print(f"  {len(cut)}/{n_quotes} eval quotes not inside any single chunk {cut or ''}")
        failed = failed or bool(over)

    if failed and chunking["unit"] == "tokens":
        sys.exit("Token budget violated: see the chunks listed above.")


if __name__ == "__main__":
    main()

"""Read the parsed corpus (data/processed/<paper_id>.json) produced by ragpapers.ingest.parse."""

import json
from collections.abc import Iterator
from pathlib import Path


def load_paper(processed_dir: str | Path, paper_id: str) -> dict:
    path = Path(processed_dir) / f"{paper_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(processed_dir: str | Path, paper_ids: list[str]) -> dict[str, dict]:
    """Map paper_id -> parsed paper. Raises FileNotFoundError if a paper was not parsed yet."""
    return {paper_id: load_paper(processed_dir, paper_id) for paper_id in paper_ids}


def iter_blocks(paper: dict) -> Iterator[tuple[dict, dict]]:
    """Yield (section, block) pairs in document order."""
    for section in paper["sections"]:
        for block in section["blocks"]:
            yield section, block

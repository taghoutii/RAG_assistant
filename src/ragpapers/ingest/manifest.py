"""Read the curated corpus manifest (data/corpus_manifest.yaml)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_FIELDS = ("arxiv_id", "version", "title", "year", "topic")


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    version: int
    title: str
    year: int
    topic: str

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}"

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.versioned_id}"


def load_manifest(path: str | Path) -> list[Paper]:
    """Load and validate the manifest. Raises ValueError on missing fields or duplicate IDs."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    papers = []
    seen = set()
    for i, entry in enumerate(data["papers"]):
        missing = [k for k in REQUIRED_FIELDS if k not in entry]
        if missing:
            raise ValueError(f"Manifest entry {i} is missing fields: {missing}")
        arxiv_id = str(entry["arxiv_id"])
        if arxiv_id in seen:
            raise ValueError(f"Duplicate arxiv_id in manifest: {arxiv_id}")
        seen.add(arxiv_id)
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                version=int(entry["version"]),
                title=entry["title"],
                year=int(entry["year"]),
                topic=entry["topic"],
            )
        )
    return papers

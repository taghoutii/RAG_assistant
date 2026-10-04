"""Parse every downloaded paper in data/raw/ into data/processed/<id>.json (+ a .md preview)."""

import json
from pathlib import Path

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.ingest.html_parser import parse_latexml_html
from ragpapers.ingest.manifest import Paper, load_manifest

# Sections that have no HTML anchor of their own link to the arXiv abstract page instead.
SECTIONS_WITHOUT_ANCHOR = {"abstract", "front", ""}


def parse_paper(paper: Paper, raw_dir: Path) -> dict:
    meta = json.loads((raw_dir / f"{paper.arxiv_id}.meta.json").read_text(encoding="utf-8"))
    html = (raw_dir / f"{paper.arxiv_id}.html").read_text(encoding="utf-8")
    sections = parse_latexml_html(html)

    for section in sections:
        if section["section_id"] in SECTIONS_WITHOUT_ANCHOR:
            section["url"] = paper.abs_url
        else:
            section["url"] = f"{meta['url']}#{section['section_id']}"

    return {
        "paper_id": paper.arxiv_id,
        "version": paper.version,
        "title": paper.title,
        "year": paper.year,
        "topic": paper.topic,
        "source": meta["source"],
        "source_url": paper.abs_url,
        "sections": sections,
    }


def to_markdown(doc: dict) -> str:
    """Human-readable preview, used to check parsing quality by eye."""
    lines = [
        f"# {doc['title']} ({doc['year']})",
        f"<{doc['source_url']}> — source: {doc['source']}",
    ]
    for section in doc["sections"]:
        flag = " [appendix]" if section["is_appendix"] else ""
        lines.append(f"\n## {' > '.join(section['path'])}{flag}")
        for block in section["blocks"]:
            if block["type"] in {"table", "code", "equation"}:
                lines.append(f"\n[{block['type']}]\n```\n{block['text']}\n```")
            elif block["type"] == "figure":
                lines.append(f"\n[figure] {block['text']}")
            else:
                lines.append(f"\n{block['text']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    config = load_config()
    raw_dir = REPO_ROOT / config["paths"]["raw_dir"]
    out_dir = REPO_ROOT / config["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = load_manifest(REPO_ROOT / config["paths"]["manifest"])
    for i, paper in enumerate(papers, start=1):
        doc = parse_paper(paper, raw_dir)
        (out_dir / f"{paper.arxiv_id}.json").write_text(
            json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / f"{paper.arxiv_id}.md").write_text(to_markdown(doc), encoding="utf-8")
        n_sections = len(doc["sections"])
        n_blocks = sum(len(s["blocks"]) for s in doc["sections"])
        print(f"[{i}/{len(papers)}] {paper.arxiv_id}: {n_sections} sections, {n_blocks} blocks")


if __name__ == "__main__":
    main()

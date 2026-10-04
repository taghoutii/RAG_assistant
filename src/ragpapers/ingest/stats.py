"""Corpus statistics and parsing-artifact checks over data/processed/*.json.

Writes reports/corpus_stats.json and prints a summary.
"""

import json
import re
import statistics
from collections import Counter

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.ingest.manifest import load_manifest

# Patterns that suggest something went wrong during parsing.
ARTIFACT_PATTERNS = {
    "latexml_class_leak": re.compile(r"ltx_"),
    "unresolved_ref": re.compile(r"\?\?"),
    "raw_latex_command_outside_math": re.compile(r"(?<!\$)\\(cite|ref|label|textbf|emph)\{"),
    "html_tag_leak": re.compile(r"</?(span|div|math|mrow|mi)\b"),
    "bibliography_leak": re.compile(r"\bIn Proceedings of\b.*\b(19|20)\d\d\b"),
}


def words(text: str) -> int:
    return len(text.split())


def paper_stats(doc: dict) -> dict:
    blocks = [(s, b) for s in doc["sections"] for b in s["blocks"]]
    main_words = sum(words(b["text"]) for s, b in blocks if not s["is_appendix"])
    appendix_words = sum(words(b["text"]) for s, b in blocks if s["is_appendix"])
    artifacts = Counter()
    for _, b in blocks:
        for name, pattern in ARTIFACT_PATTERNS.items():
            artifacts[name] += len(pattern.findall(b["text"]))
    return {
        "paper_id": doc["paper_id"],
        "title": doc["title"],
        "topic": doc["topic"],
        "source": doc["source"],
        "sections": len(doc["sections"]),
        "blocks_by_type": dict(Counter(b["type"] for _, b in blocks)),
        "words_main": main_words,
        "words_appendix": appendix_words,
        "words_total": main_words + appendix_words,
        "longest_block_words": max((words(b["text"]) for _, b in blocks), default=0),
        "artifacts": {k: v for k, v in artifacts.items() if v},
    }


def main() -> None:
    config = load_config()
    processed = REPO_ROOT / config["paths"]["processed_dir"]
    papers = load_manifest(REPO_ROOT / config["paths"]["manifest"])
    per_paper = []
    for paper in papers:
        doc = json.loads((processed / f"{paper.arxiv_id}.json").read_text(encoding="utf-8"))
        per_paper.append(paper_stats(doc))

    totals = [p["words_total"] for p in per_paper]
    block_types = Counter()
    for p in per_paper:
        block_types.update(p["blocks_by_type"])
    summary = {
        "papers": len(per_paper),
        "sources": dict(Counter(p["source"] for p in per_paper)),
        "papers_per_topic": dict(Counter(p["topic"] for p in per_paper)),
        "sections": sum(p["sections"] for p in per_paper),
        "blocks_by_type": dict(block_types),
        "words_total": sum(totals),
        "words_appendix": sum(p["words_appendix"] for p in per_paper),
        "words_per_paper": {
            "min": min(totals),
            "median": int(statistics.median(totals)),
            "max": max(totals),
        },
        "papers_with_artifacts": {
            p["paper_id"]: p["artifacts"] for p in per_paper if p["artifacts"]
        },
    }

    out = REPO_ROOT / "reports" / "corpus_stats.json"
    report = json.dumps({"summary": summary, "papers": per_paper}, indent=1) + "\n"
    out.write_text(report, encoding="utf-8", newline="\n")  # LF, as enforced by pre-commit

    print(json.dumps(summary, indent=1))
    print("\npaper_id     words(main+appx)  appx%  sections  tables  longest_block  title")
    for p in sorted(per_paper, key=lambda p: -p["words_total"]):
        appx = 100 * p["words_appendix"] / max(p["words_total"], 1)
        print(
            f"{p['paper_id']:<12} {p['words_total']:>7}  {appx:>14.0f}%  {p['sections']:>8}  "
            f"{p['blocks_by_type'].get('table', 0):>6}  {p['longest_block_words']:>13}  "
            f"{p['title'][:45]}"
        )


if __name__ == "__main__":
    main()

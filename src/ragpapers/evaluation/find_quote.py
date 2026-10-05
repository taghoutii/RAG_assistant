"""Search the parsed corpus for a phrase, to copy exact evidence quotes into the eval set.

This is a writing aid, not the retrieval system: it scans every block with fuzzy matching and
prints where the phrase appears, with surrounding context taken from the original text.

"""

import argparse

from rapidfuzz import fuzz

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.corpus import iter_blocks, load_corpus
from ragpapers.evaluation.matching import quote_score
from ragpapers.ingest.manifest import load_manifest

CONTEXT_CHARS = 200


def search(phrase: str, corpus: dict[str, dict], top: int) -> list[dict]:
    """Best-scoring blocks for the phrase, highest score first."""
    hits = []
    for paper_id, paper in corpus.items():
        for section, block in iter_blocks(paper):
            score = quote_score(phrase, block["text"])
            if score >= 60:
                hits.append(
                    {
                        "score": score,
                        "paper_id": paper_id,
                        "title": paper["title"],
                        "path": " > ".join(section["path"]),
                        "type": block["type"],
                        "text": block["text"],
                    }
                )
    hits.sort(key=lambda h: -h["score"])
    return hits[:top]


def context(phrase: str, text: str) -> str:
    """The original text around the best match (lowercasing keeps character positions)."""
    alignment = fuzz.partial_ratio_alignment(phrase.lower(), text.lower())
    start = max(alignment.dest_start - CONTEXT_CHARS, 0)
    end = min(alignment.dest_end + CONTEXT_CHARS, len(text))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Find a phrase in the parsed corpus.")
    parser.add_argument("phrase")
    parser.add_argument("--paper", help="limit the search to one arXiv id")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    paper_ids = [p.arxiv_id for p in load_manifest(REPO_ROOT / config["paths"]["manifest"])]
    if args.paper:
        paper_ids = [args.paper]
    corpus = load_corpus(REPO_ROOT / config["paths"]["processed_dir"], paper_ids)

    hits = search(args.phrase, corpus, args.top)
    if not hits:
        print("No match with score >= 60.")
    for hit in hits:
        print(f"\n[{hit['score']:.0f}] {hit['paper_id']} | {hit['title']}")
        print(f"     {hit['path']}  ({hit['type']})")
        print(f"     {context(args.phrase, hit['text'])}")


if __name__ == "__main__":
    main()

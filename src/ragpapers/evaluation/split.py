"""Freeze a stratified dev/test split of the evaluation set into eval/splits.json.

Each question type is shuffled with a fixed seed and `test_fraction` of it goes to the test set,
so both sets keep the same factual/comparison/unanswerable mix.

The split is created ONCE, after the questions are final, and is then never changed: the test set
is only used for the final report. The script refuses to overwrite an existing split unless
--force is given.

"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.corpus import load_corpus
from ragpapers.evaluation.questions import load_questions, validate
from ragpapers.ingest.manifest import load_manifest


def stratified_split(questions: list[dict], test_fraction: float, seed: int) -> dict[str, list]:
    """Return {"dev": [ids], "test": [ids]}, split per question type."""
    by_type = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q["id"])

    rng = random.Random(seed)
    dev, test = [], []
    for qtype in sorted(by_type):
        ids = sorted(by_type[qtype])
        rng.shuffle(ids)
        n_test = round(len(ids) * test_fraction)
        test += ids[:n_test]
        dev += ids[n_test:]
    return {"dev": sorted(dev), "test": sorted(test)}


def load_split(path: str | Path, name: str) -> set[str]:
    """Question ids of the "dev" or "test" split."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(data[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the frozen dev/test split.")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args()

    config = load_config()
    out = REPO_ROOT / config["paths"]["eval_splits"]
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists. The split is frozen; use --force only if you mean it.")

    questions = load_questions(REPO_ROOT / config["paths"]["eval_questions"])
    paper_ids = {p.arxiv_id for p in load_manifest(REPO_ROOT / config["paths"]["manifest"])}
    corpus = load_corpus(REPO_ROOT / config["paths"]["processed_dir"], sorted(paper_ids))
    issues = validate(questions, paper_ids, config["eval"], config["abstention_text"], corpus)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        sys.exit(f"{len(errors)} validation errors: fix them before splitting.")

    split = stratified_split(questions, config["eval"]["test_fraction"], config["seed"])
    types = {q["id"]: q["type"] for q in questions}
    record = {
        "seed": config["seed"],
        "test_fraction": config["eval"]["test_fraction"],
        "n_questions": len(questions),
        **split,
    }
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")

    for name in ("dev", "test"):
        counts = {t: sum(types[i] == t for i in split[name]) for t in sorted(set(types.values()))}
        print(f"{name}: {len(split[name])} questions {counts}")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()

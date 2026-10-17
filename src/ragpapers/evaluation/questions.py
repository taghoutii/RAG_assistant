"""Load and validate the evaluation set (eval/questions.jsonl)."""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.corpus import load_corpus
from ragpapers.evaluation.matching import locate_quote, normalize_for_matching
from ragpapers.ingest.manifest import load_manifest

TYPES = ("factual", "comparison", "unanswerable")
REQUIRED_KEYS = {"id", "question", "type", "reference_answer", "evidence"}
OPTIONAL_KEYS = {"notes"}
ID_PATTERN = re.compile(r"^q\d{3}$")
EVIDENCE_KEYS = {"paper_id", "quote"}
OPTIONAL_EVIDENCE_KEYS = {"alternatives"}  # the same fact stated elsewhere in the same paper


@dataclass
class Issue:
    level: str  # "error" or "warning"
    question_id: str
    message: str


def load_questions(path: str | Path) -> list[dict]:
    """Parse a JSONL file. Blank lines are ignored; invalid JSON raises with its line number."""
    questions = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"{path}, line {line_number}: invalid JSON ({err.msg})") from err
        if not isinstance(entry, dict):
            raise ValueError(f"{path}, line {line_number}: expected a JSON object")
        questions.append(entry)
    return questions


def evidence_groups(q: dict, with_alternatives: bool = True) -> list[list[str]]:
    """One group of quotes per evidence item: the original quote, then its alternatives.

    A retrieved chunk is relevant for an item if it contains any quote of the item's group
    (evaluation/metrics.py). `with_alternatives=False` gives the original single-quote groups.
    """
    groups = []
    for ev in q.get("evidence") or []:
        alternatives = ev.get("alternatives", []) if with_alternatives else []
        groups.append([ev["quote"], *alternatives])
    return groups


# --- per-question checks -----------------------------------------------------------------------


def check_schema(q: dict, paper_ids: set[str], cfg: dict, abstention_text: str) -> list[Issue]:
    """Field presence, types and type-specific evidence rules. Needs no corpus."""
    qid = str(q.get("id", "?"))
    issues = []

    def error(msg):
        issues.append(Issue("error", qid, msg))

    def warning(msg):
        issues.append(Issue("warning", qid, msg))

    missing = REQUIRED_KEYS - q.keys()
    unknown = q.keys() - REQUIRED_KEYS - OPTIONAL_KEYS
    if missing:
        error(f"missing fields: {sorted(missing)}")
    if unknown:
        error(f"unknown fields (typo?): {sorted(unknown)}")
    if missing:
        return issues

    if not ID_PATTERN.match(str(q["id"])):
        error(f"id {q['id']!r} must look like q001")
    if not isinstance(q["question"], str) or not q["question"].strip():
        error("question must be a non-empty string")
    elif not q["question"].strip().endswith("?"):
        warning("question does not end with '?'")
    if q["type"] not in TYPES:
        error(f"type {q['type']!r} must be one of {TYPES}")
        return issues
    if not isinstance(q["reference_answer"], str) or not q["reference_answer"].strip():
        error("reference_answer must be a non-empty string")

    evidence = q["evidence"]
    if not isinstance(evidence, list):
        error("evidence must be a list")
        return issues

    for i, ev in enumerate(evidence):
        if (
            not isinstance(ev, dict)
            or not ev.keys() >= EVIDENCE_KEYS
            or ev.keys() - EVIDENCE_KEYS - OPTIONAL_EVIDENCE_KEYS
        ):
            error(
                f"evidence[{i}] must be an object with 'paper_id' and 'quote' "
                "(and optionally 'alternatives')"
            )
            continue
        issues += _check_alternatives(qid, i, ev, cfg)
        if ev["paper_id"] not in paper_ids:
            error(f"evidence[{i}]: paper_id {ev['paper_id']!r} is not in the corpus manifest")
        n_words = len(str(ev["quote"]).split())
        if n_words == 0:
            error(f"evidence[{i}]: quote is empty")
        elif n_words < cfg["quote_min_words"]:
            warning(f"evidence[{i}]: quote has {n_words} words (min {cfg['quote_min_words']})")
        elif n_words > cfg["quote_max_words"]:
            warning(f"evidence[{i}]: quote has {n_words} words (max {cfg['quote_max_words']})")

    papers = {ev.get("paper_id") for ev in evidence if isinstance(ev, dict)}
    if q["type"] == "factual":
        if not evidence:
            error("factual question needs at least one evidence quote")
        elif len(papers) > 1:
            error(f"factual question cites {len(papers)} papers; use type 'comparison'")
    elif q["type"] == "comparison":
        if len(papers) < 2:
            error("comparison question needs evidence from at least 2 different papers")
    elif q["type"] == "unanswerable":
        if evidence:
            error("unanswerable question must have empty evidence []")
        if q["reference_answer"].strip() != abstention_text:
            error(f"unanswerable reference_answer must be exactly: {abstention_text!r}")
    return issues


def _check_alternatives(qid: str, i: int, ev: dict, cfg: dict) -> list[Issue]:
    """Alternatives: a list of at most `alternatives_max` distinct quotes of 5-50 words."""
    if "alternatives" not in ev:
        return []
    alternatives = ev["alternatives"]
    if not isinstance(alternatives, list) or not all(isinstance(a, str) for a in alternatives):
        return [Issue("error", qid, f"evidence[{i}]: alternatives must be a list of strings")]
    issues = []
    if len(alternatives) > cfg["alternatives_max"]:
        issues.append(
            Issue(
                "error",
                qid,
                f"evidence[{i}]: {len(alternatives)} alternatives (max {cfg['alternatives_max']})",
            )
        )
    seen = {normalize_for_matching(str(ev.get("quote", "")))}
    for j, alt in enumerate(alternatives):
        n_words = len(alt.split())
        if not cfg["quote_min_words"] <= n_words <= cfg["quote_max_words"]:
            issues.append(
                Issue(
                    "error",
                    qid,
                    f"evidence[{i}].alternatives[{j}]: {n_words} words (must be "
                    f"{cfg['quote_min_words']}-{cfg['quote_max_words']})",
                )
            )
        if normalize_for_matching(alt) in seen:
            issues.append(Issue("error", qid, f"evidence[{i}].alternatives[{j}]: duplicate quote"))
        seen.add(normalize_for_matching(alt))
    return issues


def check_quotes(q: dict, corpus: dict[str, dict], threshold: float) -> list[Issue]:
    """Each quote (and alternative) must be found verbatim in one block of its own paper."""
    qid = str(q.get("id", "?"))
    issues = []
    for i, ev in enumerate(q.get("evidence") or []):
        if not isinstance(ev, dict) or ev.get("paper_id") not in corpus or not ev.get("quote"):
            continue  # already reported by check_schema
        paper_id = ev["paper_id"]
        issues += _check_one_quote(qid, f"evidence[{i}]", ev["quote"], paper_id, corpus, threshold)
        alternatives = ev.get("alternatives", [])
        if not isinstance(alternatives, list):
            continue  # already reported by check_schema
        for j, alt in enumerate(alternatives):
            if isinstance(alt, str) and alt.strip():
                where = f"evidence[{i}].alternatives[{j}]"
                issues += _check_one_quote(qid, where, alt, paper_id, corpus, threshold)
    return issues


def _check_one_quote(
    qid: str, label: str, quote: str, paper_id: str, corpus: dict, threshold: float
) -> list[Issue]:
    """One quote must be found verbatim (after normalization) in one block of `paper_id`."""
    match = locate_quote(quote, corpus[paper_id])
    if match is not None and match.exact:
        return []
    score = match.score if match else 0.0
    where = " > ".join(match.section_path) if match else "-"
    if score >= threshold:
        message = (
            f"{label}: only a fuzzy match (score {score:.0f}) in [{where}]. "
            "Copy the exact text with: python -m ragpapers.evaluation.find_quote"
        )
        return [Issue("warning", qid, message)]
    other = _found_in_other_paper(quote, paper_id, corpus, threshold)
    hint = f" It does appear in paper {other}." if other else ""
    message = f"{label}: quote not found in {paper_id} (best score {score:.0f} in [{where}]).{hint}"
    return [Issue("error", qid, message)]


def _found_in_other_paper(quote: str, paper_id: str, corpus: dict, threshold: float) -> str | None:
    for other_id, paper in corpus.items():
        if other_id == paper_id:
            continue
        match = locate_quote(quote, paper)
        if match is not None and match.score >= threshold:
            return other_id
    return None


# --- whole-set checks --------------------------------------------------------------------------


def check_set(questions: list[dict], cfg: dict) -> list[Issue]:
    """Unique ids, duplicate questions, set size and type mix."""
    issues = []
    ids = Counter(str(q.get("id")) for q in questions)
    for qid, count in ids.items():
        if count > 1:
            issues.append(Issue("error", qid, f"id used {count} times"))

    texts = [
        (str(q.get("id")), normalize_for_matching(str(q.get("question", "")))) for q in questions
    ]
    for i, (id_a, text_a) in enumerate(texts):
        for id_b, text_b in texts[i + 1 :]:
            if text_a == text_b:
                issues.append(Issue("error", id_b, f"same question as {id_a}"))
            elif fuzz.ratio(text_a, text_b) >= cfg["near_duplicate_threshold"]:
                issues.append(Issue("warning", id_b, f"near-duplicate of {id_a}"))

    low, high = cfg["target_size"]
    if not low <= len(questions) <= high:
        issues.append(Issue("warning", "set", f"{len(questions)} questions (target {low}-{high})"))

    counts = Counter(q.get("type") for q in questions)
    for t, target in cfg["target_mix"].items():
        share = counts[t] / len(questions) if questions else 0.0
        if abs(share - target) > cfg["mix_tolerance"]:
            issues.append(
                Issue("warning", "set", f"{t}: {share:.0%} of questions (target {target:.0%})")
            )
    return issues


def validate(
    questions: list[dict],
    paper_ids: set[str],
    cfg: dict,
    abstention_text: str,
    corpus: dict[str, dict] | None,
) -> list[Issue]:
    """All checks. Pass corpus=None to skip the quote checks (e.g. in CI, where data is absent)."""
    issues = []
    for q in questions:
        issues += check_schema(q, paper_ids, cfg, abstention_text)
        if corpus is not None:
            issues += check_quotes(q, corpus, cfg["quote_fuzzy_threshold"])
    return issues + check_set(questions, cfg)


# --- command line ------------------------------------------------------------------------------


def questions_per_paper(questions: list[dict], paper_ids: list[str]) -> dict[str, int]:
    """Number of questions citing each paper (a comparison question counts for each of its papers).

    Every paper in `paper_ids` appears, with 0 if no question cites it.
    """
    counts = dict.fromkeys(paper_ids, 0)
    for q in questions:
        cited = {ev.get("paper_id") for ev in q.get("evidence") or [] if isinstance(ev, dict)}
        for paper_id in cited:
            if paper_id in counts:
                counts[paper_id] += 1
    return counts


def print_report(questions: list[dict], issues: list[Issue], papers: list) -> None:
    counts = Counter(q.get("type") for q in questions)
    total = len(questions) or 1
    print(f"{len(questions)} questions")
    for t in TYPES:
        print(f"  {t:<13} {counts[t]:>3}  ({counts[t] / total:.0%})")
    n_quotes = sum(len(q.get("evidence") or []) for q in questions)
    print(f"  evidence quotes: {n_quotes}")

    per_paper = questions_per_paper(questions, [p.arxiv_id for p in papers])
    uncovered = sum(1 for n in per_paper.values() if n == 0)
    print(f"\nQuestions per paper ({uncovered} of {len(papers)} papers have none)")
    for paper in papers:
        n = per_paper[paper.arxiv_id]
        marker = "  <- none" if n == 0 else ""
        print(f"  {paper.arxiv_id:<11} {n:>2}  [{paper.topic}] {paper.title[:50]}{marker}")

    for level in ("error", "warning"):
        selected = [i for i in issues if i.level == level]
        print(f"\n{len(selected)} {level}s")
        for issue in selected:
            print(f"  [{issue.question_id}] {issue.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an evaluation question file.")
    parser.add_argument("path", nargs="?", help="JSONL file (default: eval/questions.jsonl)")
    parser.add_argument("--skip-quotes", action="store_true", help="schema checks only")
    args = parser.parse_args()

    config = load_config()
    path = Path(args.path) if args.path else REPO_ROOT / config["paths"]["eval_questions"]
    papers = load_manifest(REPO_ROOT / config["paths"]["manifest"])
    paper_ids = {p.arxiv_id for p in papers}
    corpus = None
    if not args.skip_quotes:
        corpus = load_corpus(REPO_ROOT / config["paths"]["processed_dir"], sorted(paper_ids))

    questions = load_questions(path)
    issues = validate(questions, paper_ids, config["eval"], config["abstention_text"], corpus)
    print(f"Validating {path}")
    print_report(questions, issues, papers)
    if any(i.level == "error" for i in issues):
        sys.exit(1)


if __name__ == "__main__":
    main()

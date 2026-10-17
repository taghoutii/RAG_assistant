"""Alternative evidence quotes: propose candidates from the papers, apply the approved ones.

A paper often states the same fact in more than one place (abstract, introduction, results).
An evidence item with one quote then marks a chunk that states the fact in other words as
irrelevant. Alternatives fix this: each evidence item may carry up to `alternatives_max` extra
quotes from the SAME paper, and it counts as found where any of them is (evaluation/metrics.py).

Candidates are chosen blind to retrieval: they come only from the evidence item's own parsed
paper and the question's reference answer, never from any retriever's output, ranks or scores.
A sentence is a candidate if:
- it is in a paragraph, list or figure caption (no tables, no equations), and has >= 5 words;
- it shares >= `min_shared_words` content words with the reference answer AND at least
  `min_shared_fraction` of the answer's content words (content words = the BM25 tokenizer:
  lowercase alphanumerics without stopwords);
- it is not (part of) the original quote.
Candidates are listed in reading order, with no cap.

Usage (PowerShell, from the repo root):
    python -m ragpapers.evaluation.alternates candidates   # writes eval/alternates/batch_XX.md
    # tick "- [x]" the candidates that state the same fact; trim the text in place if needed
    python -m ragpapers.evaluation.alternates apply        # adds them to eval/questions.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.corpus import iter_blocks, load_corpus
from ragpapers.evaluation.matching import locate_quote, quote_score
from ragpapers.evaluation.questions import load_questions
from ragpapers.index.bm25 import tokenize

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$])")
CANDIDATE_BLOCKS = ("paragraph", "list", "figure")
CHECKBOX = re.compile(r"^- \[(?P<mark>[ xX])\] `(?P<cid>q\d{3}\.e\d+\.c\d+)`")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(" ".join(text.split())) if s.strip()]


def candidate_sentences(paper: dict, reference_answer: str, quote: str, cfg: dict) -> list[dict]:
    """Sentences of `paper` that may restate the fact, in reading order (see module docstring)."""
    answer_words = set(tokenize(reference_answer))
    out = []
    for section, block in iter_blocks(paper):
        if block["type"] not in CANDIDATE_BLOCKS:
            continue
        sentences = split_sentences(block["text"])
        for k, sentence in enumerate(sentences):
            n_words = len(sentence.split())
            if n_words < 5:
                continue
            shared = answer_words & set(tokenize(sentence))
            if len(shared) < cfg["min_shared_words"]:
                continue
            if len(shared) < cfg["min_shared_fraction"] * len(answer_words):
                continue
            threshold = cfg["same_text_threshold"]
            if (
                quote_score(quote, sentence) >= threshold
                or quote_score(sentence, quote) >= threshold
            ):
                continue  # the original quote itself (or a piece of it)
            out.append(
                {
                    "section": " > ".join(section["path"]),
                    "text": sentence,
                    "n_words": n_words,
                    "shared": sorted(shared),
                    "next": sentences[k + 1] if k + 1 < len(sentences) else "",
                }
            )
    return out


def build_candidates(questions: list[dict], corpus: dict, cfg: dict) -> dict[str, dict]:
    """{candidate id: {...}} for every answerable question; id = q101.e0.c3."""
    candidates = {}
    for q in questions:
        for e, ev in enumerate(q["evidence"]):
            paper = corpus[ev["paper_id"]]
            found = candidate_sentences(paper, q["reference_answer"], ev["quote"], cfg)
            for c, cand in enumerate(found):
                cid = f"{q['id']}.e{e}.c{c}"
                candidates[cid] = {
                    "question": q["id"],
                    "evidence": e,
                    "paper_id": ev["paper_id"],
                    "quote": ev["quote"],
                    **cand,
                }
    return candidates


def render_batch(questions: list[dict], candidates: dict, corpus: dict, max_words: int) -> str:
    lines = [
        "# Alternative quotes: candidates for approval",
        "",
        "Tick `- [x]` only if the sentence states the SAME fact as the original quote (same "
        "paper is guaranteed). You may trim the text on the `>` line, or extend it into the next "
        "sentence shown below it, to a 5-50 word span; it must stay verbatim (one paragraph). "
        "At most 3 alternatives per evidence item. Then run "
        "`python -m ragpapers.evaluation.alternates apply`.",
    ]
    for q in questions:
        lines += [
            "",
            f"## {q['id']} ({q['type']})",
            "",
            f"**Q:** {q['question']}",
            "",
            f"**Reference answer:** {q['reference_answer']}",
        ]
        for e, ev in enumerate(q["evidence"]):
            mine = [(cid, c) for cid, c in candidates.items() if cid.startswith(f"{q['id']}.e{e}.")]
            existing = ev.get("alternatives", [])
            lines += [
                "",
                f"### Evidence {e}: {corpus[ev['paper_id']]['title']} ({ev['paper_id']})",
                "",
                f'Original quote: "{ev["quote"]}"',
            ]
            if existing:
                lines.append(f"Already approved: {len(existing)}")
            if not mine:
                lines += ["", "_No candidates._"]
            for cid, c in mine:
                trim = (
                    f" **(> {max_words} words: trim before ticking)**"
                    if c["n_words"] > max_words
                    else ""
                )
                lines += [
                    "",
                    f"- [ ] `{cid}` ({c['n_words']} words) {c['section']}{trim}",
                    f"  > {c['text']}",
                ]
                if c["next"]:
                    lines.append(
                        f"  _next sentence (context; you may extend into it): {c['next']}_"
                    )
    return "\n".join(lines) + "\n"


def read_ticked(batch_text: str) -> dict[str, str]:
    """{candidate id: text} for every ticked candidate (the text as possibly trimmed)."""
    ticked = {}
    lines = batch_text.splitlines()
    for i, line in enumerate(lines):
        m = CHECKBOX.match(line)
        if m and m["mark"] in "xX":
            text_line = lines[i + 1] if i + 1 < len(lines) else ""
            if not text_line.lstrip().startswith(">"):
                raise ValueError(f"{m['cid']}: the '>' text line after the checkbox is missing")
            ticked[m["cid"]] = text_line.lstrip()[1:].strip()
    return ticked


def apply_ticked(
    questions: list[dict], ticked: dict[str, str], candidates: dict, corpus: dict, cfg: dict
) -> list[str]:
    """Add ticked candidates as alternatives (in place). Returns a list of problems (empty = ok).

    Nothing is changed if any problem is found.
    """
    problems = []
    by_id = {q["id"]: q for q in questions}
    additions: dict[tuple[str, int], list[str]] = {}
    for cid, text in sorted(ticked.items()):
        cand = candidates.get(cid)
        if cand is None:
            problems.append(f"{cid}: unknown candidate id")
            continue
        n_words = len(text.split())
        if not cfg["quote_min_words"] <= n_words <= cfg["quote_max_words"]:
            problems.append(f"{cid}: {n_words} words (must be 5-50: trim it)")
        match = locate_quote(text, corpus[cand["paper_id"]])
        if match is None or not match.exact:
            problems.append(
                f"{cid}: text is not verbatim in {cand['paper_id']} (edited by mistake?)"
            )
        additions.setdefault((cand["question"], cand["evidence"]), []).append(text)

    for (qid, e), texts in additions.items():
        ev = by_id[qid]["evidence"][e]
        merged = ev.get("alternatives", []) + [
            t for t in texts if t not in ev.get("alternatives", [])
        ]
        if len(merged) > cfg["alternatives_max"]:
            problems.append(
                f"{qid} evidence {e}: {len(merged)} alternatives (max {cfg['alternatives_max']})"
            )
    if problems:
        return problems

    for (qid, e), texts in additions.items():
        ev = by_id[qid]["evidence"][e]
        ev["alternatives"] = ev.get("alternatives", []) + [
            t for t in texts if t not in ev.get("alternatives", [])
        ]
    return []


def write_questions(path: Path, questions: list[dict]) -> None:
    """Same format as the hand-written file: one compact JSON object per line, UTF-8."""
    text = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternative evidence quotes.")
    parser.add_argument("command", choices=["candidates", "apply"])
    args = parser.parse_args()

    config = load_config()
    cfg = {**config["eval"], **config["alternates"]}
    q_path = REPO_ROOT / config["paths"]["eval_questions"]
    out_dir = REPO_ROOT / config["alternates"]["dir"]
    questions = load_questions(q_path)
    answerable = [q for q in questions if q["evidence"]]
    paper_ids = sorted({ev["paper_id"] for q in answerable for ev in q["evidence"]})
    corpus = load_corpus(REPO_ROOT / config["paths"]["processed_dir"], paper_ids)
    candidates = build_candidates(answerable, corpus, cfg)

    if args.command == "candidates":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidates.json").write_text(
            json.dumps(candidates, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        size = cfg["batch_size"]
        for b in range(0, len(answerable), size):
            batch = answerable[b : b + size]
            path = out_dir / f"batch_{b // size + 1:02d}.md"
            if path.exists():
                print(f"{path.name} exists, not overwritten (it may hold your ticks)")
                continue
            path.write_text(
                render_batch(batch, candidates, corpus, cfg["quote_max_words"]),
                encoding="utf-8",
                newline="\n",
            )
            n = sum(c["question"] in {q["id"] for q in batch} for c in candidates.values())
            print(f"{path.name}: {len(batch)} questions, {n} candidates")
        print(f"{len(candidates)} candidates for {len(answerable)} answerable questions")
        return

    saved = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
    if saved != candidates:
        sys.exit("candidates.json is out of date (corpus, questions or filter changed).")
    ticked = {}
    for path in sorted(out_dir.glob("batch_*.md")):
        ticked.update(read_ticked(path.read_text(encoding="utf-8")))
    problems = apply_ticked(questions, ticked, candidates, corpus, cfg)
    if problems:
        print("Nothing applied:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    write_questions(q_path, questions)
    print(
        f"Applied {len(ticked)} ticked alternatives. "
        "Now run: python -m ragpapers.evaluation.questions"
    )


if __name__ == "__main__":
    main()

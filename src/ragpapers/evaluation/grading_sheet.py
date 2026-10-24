"""The user's blind grading sheet for the 20 judge-validation answers, and the grades file.

Writes:
    eval/grading/sheet.md     what to read (gitignored: it contains corpus text)
    eval/human_grades.jsonl   where the grades go: one line per answer, labels null until
                              filled in with 0 or 1 (never overwritten once it exists)

The sheet shows exactly what the judge sees, in two parts:
    Part 1  question + the 8 numbered passages + the answer:
            faithfulness, citation correctness, relevance (no reference answer anywhere)
    Part 2  question + the reference answer + the answer: correctness
Grade all of Part 1 before reading Part 2. No judge grade exists or is shown at this point.

Usage (PowerShell, from the repo root, after the hybrid dev answers exist):
    python -m ragpapers.evaluation.grading_sheet
"""

import json

from ragpapers.chunking.build import chunks_path, load_chunks
from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.judge import GROUNDING_METRICS, RUBRIC, answers_path, sample_ids
from ragpapers.evaluation.questions import load_questions
from ragpapers.generation.prompt import format_context
from ragpapers.generation.runlog import load_done


def grades_template(answers: list[dict]) -> str:
    lines = []
    for a in answers:
        record = {"id": a["id"], "type": a["type"], "abstained": a["abstained"]}
        record |= dict.fromkeys((*GROUNDING_METRICS, "correctness"))
        record["notes"] = ""
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def make_sheet(answers: list[dict], questions: dict, by_id: dict) -> str:
    out = [
        "# Judge validation: blind grading sheet",
        "",
        f"{len(answers)} hybrid dev answers (openai/gpt-oss-120b, k = 8). For each answer, "
        "write 0 or 1 per metric in eval/human_grades.jsonl. Abstentions need no grades.",
        "",
        "Grade all of **Part 1** before reading Part 2: Part 2 shows the reference answers.",
        "",
        "## Rubric (the judge gets the same definitions)",
        "",
        *[f"- **{m}**: {RUBRIC[m]}" for m in GROUNDING_METRICS],
        "",
        "A factual claim is a statement about what the papers say or report; a sentence that "
        "only says what the context does not cover is not a claim.",
        "",
        "---",
        "",
        "# Part 1: faithfulness, citation correctness, relevance",
    ]
    for n, a in enumerate(answers, start=1):
        q = questions[a["id"]]
        out += ["", f"## {n}. {a['id']} ({q['type']})", "", f"**Question:** {q['question']}", ""]
        out += [
            "**Context passages:**",
            "",
            "```text",
            format_context([by_id[i] for i in a["chunk_ids"]]),
            "```",
            "",
        ]
        out += ["**Answer:**", "", "```text", a["text"], "```", ""]
        if a["abstained"]:
            out.append("_Abstained: no grades needed._")
        else:
            out.append("Grade: faithfulness _, citation_correctness _, relevance _")
    out += [
        "",
        "---",
        "",
        "# Part 2: correctness (reference answers below)",
        "",
        f"- **correctness**: {RUBRIC['correctness']}",
    ]
    for n, a in enumerate(answers, start=1):
        if a["abstained"]:
            continue
        q = questions[a["id"]]
        out += ["", f"## {n}. {a['id']}", "", f"**Question:** {q['question']}", ""]
        out += [f"**Reference answer:** {q['reference_answer']}", ""]
        out += ["**Answer:**", "", "```text", a["text"], "```", "", "Grade: correctness _"]
    return "\n".join(out) + "\n"


def main() -> None:
    config = load_config()
    jv = config["judge_validation"]
    ids = sample_ids(config)
    done = load_done(answers_path(config, "dev", jv["setup"]))
    missing = [i for i in ids if i not in done]
    if missing:
        raise SystemExit(f"no {jv['setup']} dev answer yet for: {', '.join(missing)}")
    answers = [done[i] for i in ids]
    questions = {q["id"]: q for q in load_questions(REPO_ROOT / config["paths"]["eval_questions"])}
    by_id = {c["chunk_id"]: c for c in load_chunks(chunks_path(config))}

    sheet = REPO_ROOT / jv["grading_sheet"]
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(make_sheet(answers, questions, by_id), encoding="utf-8", newline="\n")
    print(f"Wrote {sheet}")

    grades = REPO_ROOT / jv["human_grades"]
    if grades.exists():
        print(f"Kept the existing {grades} (never overwritten)")
    else:
        grades.write_text(grades_template(answers), encoding="utf-8", newline="\n")
        print(f"Wrote {grades}: fill in 0/1 for each label")


if __name__ == "__main__":
    main()

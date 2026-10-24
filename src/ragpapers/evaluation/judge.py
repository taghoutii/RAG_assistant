"""LLM-as-judge: binary labels for faithfulness, citation correctness, relevance, correctness.

Each answered question gets up to two judge calls, so the reference answer never leaks into
the grounding labels:

    grounding    question + the same numbered chunks the generator saw + the answer
                 -> one entry per factual claim (cited passages, supported?, cited support?)
                    and whether the answer addresses the question
    correctness  question + reference answer + the answer (answerable questions only)
                 -> the reference's key facts, each present in the answer or not

The judge lists claims / key facts; the 0/1 labels are computed here from those lists, so
every label can be traced back to the claims behind it:

    faithfulness          1 if every claim is supported by the passages
    citation_correctness  1 if every claim cites a passage that supports it, and no cited
                          number is outside the passages shown
    relevance             1 if the answer addresses the question asked
    correctness           1 if every key fact of the reference answer is in the answer

Abstentions get no judge labels (abstention accuracy is scored by rule). Answered
unanswerable questions get the three grounding labels but no correctness label.

Blind grading: on the validation setup (hybrid, dev) the judge refuses to run until the
user's grades of the 20 sample answers are complete in eval/human_grades.jsonl.
"""

import argparse
import json
import re

from ragpapers.chunking.build import chunks_path, load_chunks
from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.questions import load_questions
from ragpapers.generation.client import DailyQuotaReached, make_client, safe_name
from ragpapers.generation.prompt import format_context
from ragpapers.generation.providers import Message
from ragpapers.generation.runlog import append_record, load_done

# What each label means, worded the same for the user's grading sheet and the judge.
RUBRIC = {
    "faithfulness": "1 if every factual claim in the answer is supported by the context "
    "passages (taken together), else 0. Use only the passages, not outside knowledge.",
    "citation_correctness": "1 if every factual claim carries a citation and the cited "
    "passage(s) support that claim, else 0. A citation to a passage number that was not "
    "shown makes it 0.",
    "relevance": "1 if the answer addresses the question that was asked, whether or not it "
    "is correct, else 0.",
    "correctness": "1 if the answer states the key facts of the reference answer (none "
    "missing or contradicted), else 0. Extra correct details do not matter.",
}
GROUNDING_METRICS = ("faithfulness", "citation_correctness", "relevance")
METRICS = (*GROUNDING_METRICS, "correctness")

GROUNDING_SYSTEM = """You grade answers produced by a question-answering system over AI/ML
research papers. The system was given numbered context passages and told to answer only from
them, citing passages as [n] (or 【n】).

You get the question, the same numbered passages, and the answer. Then:
1. Split the answer into its factual claims: each statement about what the papers say or
   report. Do not count sentences that only say what the context does not cover.
2. For each claim, give:
   - "citations": the passage numbers cited for that claim (an empty list if none);
   - "supported": true if the passages, taken together, state or directly imply the claim;
     false if it is absent from them, goes beyond them, or contradicts them. Use only the
     passages, not your own knowledge, even if the claim is true;
   - "cited_support": true if the cited passages themselves support the claim; false if no
     passage is cited or the cited ones do not support it.
3. "relevant": true if the answer addresses the question that was asked, whether or not it
   is correct; false if it answers a different question or is mostly about something else.

Reply with a JSON object only, in exactly this form:
{"claims": [{"claim": "...", "citations": [1], "supported": true, "cited_support": true}],
 "relevant": true}"""

GROUNDING_USER = """Question: {question}

Context passages:

{context}

Answer to grade:
{answer}"""

CORRECTNESS_SYSTEM = """You check an answer against a reference answer written by an expert.
1. List the key facts of the reference answer: the facts needed to answer the question.
   Ignore its wording, examples and side remarks.
2. For each key fact, "present": true if the answer states it or something equivalent;
   false if the answer omits it or contradicts it.
Extra details in the answer do not matter. Judge only against the reference answer, not your
own knowledge.

Reply with a JSON object only, in exactly this form:
{"key_facts": [{"fact": "...", "present": true}]}"""

CORRECTNESS_USER = """Question: {question}

Reference answer:
{reference}

Answer to grade:
{answer}"""


class JudgeFormatError(ValueError):
    """The judge's reply is not the JSON shape asked for."""


def grounding_messages(question: str, chunks: list[dict], answer: str) -> list[Message]:
    user = GROUNDING_USER.format(question=question, context=format_context(chunks), answer=answer)
    return [Message("system", GROUNDING_SYSTEM), Message("user", user)]


def correctness_messages(question: str, reference: str, answer: str) -> list[Message]:
    user = CORRECTNESS_USER.format(question=question, reference=reference, answer=answer)
    return [Message("system", CORRECTNESS_SYSTEM), Message("user", user)]


def parse_json(text: str) -> dict | list:
    """The JSON in a judge reply (tolerates code fences and <think> blocks).

    Normally an object. A bare list is returned as is: the label functions decide whether
    it is an acceptable shape (a bare key-facts list is; anything else is not).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")  # an object inside surrounding prose
        if start < 0 or end < start:
            raise JudgeFormatError(f"no JSON object in reply: {text[:120]!r}") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as err:
            raise JudgeFormatError(f"invalid JSON: {err}") from err
    if not isinstance(data, dict | list):
        raise JudgeFormatError("reply is not a JSON object")
    return data


def _check_items(items, name: str, fields: dict) -> list[dict]:
    if not isinstance(items, list):
        raise JudgeFormatError(f"'{name}' must be a list")
    for item in items:
        if not isinstance(item, dict):
            raise JudgeFormatError(f"each '{name}' entry must be an object")
        for field, kind in fields.items():
            if not isinstance(item.get(field), kind):
                raise JudgeFormatError(f"'{name}' entry needs '{field}' ({kind.__name__})")
    return items


def grounding_labels(parsed: dict, invalid_citations: list[int]) -> dict:
    """faithfulness, citation_correctness and relevance (0/1) from the judge's claim list."""
    if not isinstance(parsed, dict):
        raise JudgeFormatError("grounding reply must be a JSON object")
    claims = _check_items(
        parsed.get("claims"),
        "claims",
        {"claim": str, "citations": list, "supported": bool, "cited_support": bool},
    )
    if not isinstance(parsed.get("relevant"), bool):
        raise JudgeFormatError("'relevant' must be true or false")
    return {
        "faithfulness": int(all(c["supported"] for c in claims)),
        "citation_correctness": int(
            not invalid_citations and all(c["citations"] and c["cited_support"] for c in claims)
        ),
        "relevance": int(parsed["relevant"]),
        "n_claims": len(claims),
        "n_supported": sum(c["supported"] for c in claims),
    }


def correctness_label(parsed: dict | list) -> dict:
    """correctness (0/1): every key fact of the reference is present in the answer.

    Accepts {"key_facts": [...]} as asked, and the bare list [...] of key facts (a format
    slip seen from Gemini: same content, outer object missing).
    """
    facts = parsed if isinstance(parsed, list) else parsed.get("key_facts")
    facts = _check_items(facts, "key_facts", {"fact": str, "present": bool})
    if not facts:
        raise JudgeFormatError("'key_facts' is empty")
    return {
        "correctness": int(all(f["present"] for f in facts)),
        "n_key_facts": len(facts),
        "n_present": sum(f["present"] for f in facts),
    }


def _call(client, messages: list[Message], parse) -> dict:
    """One judge call; a malformed reply is recorded as an error, not raised."""
    response = client.generate(messages)
    out = {
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "finish_reason": response.finish_reason,
        "cached": response.cached,
        "request_hash": response.request_hash,
    }
    try:
        parsed = parse_json(response.text)
        out["labels"] = parse(parsed)
        out["parsed"] = parsed
    except JudgeFormatError as err:
        out["error"] = str(err)
        out["raw"] = response.text
    return out


def judge_answer(answer: dict, question: dict, chunks: list[dict], client) -> dict:
    """Judge labels for one generated answer (a record from run_generation)."""
    record = {"id": answer["id"], "type": question["type"], "abstained": answer["abstained"]}
    labels = dict.fromkeys(METRICS)
    if answer["abstained"]:
        return {**record, "labels": labels}

    invalid = answer["invalid_citations"]
    grounding = _call(
        client,
        grounding_messages(question["question"], chunks, answer["text"]),
        lambda parsed: grounding_labels(parsed, invalid),
    )
    record["grounding"] = grounding
    for metric in GROUNDING_METRICS:
        labels[metric] = grounding.get("labels", {}).get(metric)

    if question["type"] != "unanswerable":
        correctness = _call(
            client,
            correctness_messages(
                question["question"], question["reference_answer"], answer["text"]
            ),
            correctness_label,
        )
        record["correctness"] = correctness
        labels["correctness"] = correctness.get("labels", {}).get("correctness")
    return {**record, "labels": labels}


# --- files and blind-grading guard --------------------------------------------------------------


def answers_path(config: dict, split: str, setup: str):
    return (
        REPO_ROOT / config["paths"]["results_dir"] / "generation" / split / f"{setup}.answers.jsonl"
    )


def judgments_path(config: dict, split: str, setup: str, model: str):
    name = f"{setup}.judgments.{safe_name(model)}.jsonl"
    return REPO_ROOT / config["paths"]["results_dir"] / "generation" / split / name


def load_human_grades(path) -> dict[str, dict]:
    """The user's grades by question ID. Strict: a malformed line is an error, not skipped."""
    if not path.exists():
        return {}
    grades = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"{path.name} line {n}: {err}") from err
        for metric in METRICS:
            if record.get(metric) not in (None, 0, 1):
                raise ValueError(f"{path.name} line {n}: {metric} must be 0, 1 or null")
        grades[record["id"]] = record
    return grades


def missing_grades(grades: dict[str, dict], sample_ids: list[str], metrics=METRICS) -> list[str]:
    """Sample IDs whose grades are not all filled in with 0 or 1 (abstentions need none)."""
    missing = []
    for qid in sample_ids:
        g = grades.get(qid)
        if g is None or not g.get("abstained") and any(g.get(m) not in (0, 1) for m in metrics):
            missing.append(qid)
    return missing


def sample_ids(config: dict) -> list[str]:
    sample = json.loads((REPO_ROOT / config["judge_validation"]["sample"]).read_text("utf-8"))
    return [qid for ids in sample["questions"].values() for qid in ids]


def check_blind(config: dict, split: str, setup: str) -> None:
    """Refuse to judge the validation answers before the user's grades are recorded."""
    jv = config["judge_validation"]
    if split != "dev" or setup != jv["setup"]:
        return
    grades = load_human_grades(REPO_ROOT / jv["human_grades"])
    missing = missing_grades(grades, sample_ids(config), jv["metrics"])
    if missing:
        raise SystemExit(
            f"Blind grading: {len(missing)} of the 20 sample answers are not graded yet in "
            f"{jv['human_grades']} ({', '.join(missing[:5])}...). The judge runs after."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge generated answers (LLM-as-judge).")
    parser.add_argument("--setup", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--run", action="store_true", help="make the calls (default: estimate)")
    parser.add_argument(
        "--sample-only", action="store_true", help="only the 20 judge-validation answers"
    )
    args = parser.parse_args(argv)

    config = load_config()
    check_blind(config, args.split, args.setup)
    answers = list(load_done(answers_path(config, args.split, args.setup)).values())
    if not answers:
        raise SystemExit("no answers yet: run ragpapers.evaluation.run_generation first")
    questions = {q["id"]: q for q in load_questions(REPO_ROOT / config["paths"]["eval_questions"])}
    by_id = {c["chunk_id"]: c for c in load_chunks(chunks_path(config))}
    client = make_client(config, "judge")
    out = judgments_path(config, args.split, args.setup, client.model)
    done = load_done(out)

    todo = [a for a in answers if a["id"] not in done]
    if args.sample_only:
        keep = set(sample_ids(config))
        todo = [a for a in todo if a["id"] in keep]
    calls, tokens = 0, 0.0
    for a in todo:
        if a["abstained"]:
            continue
        q, chunks = questions[a["id"]], [by_id[i] for i in a["chunk_ids"]]
        msgs = [grounding_messages(q["question"], chunks, a["text"])]
        if q["type"] != "unanswerable":
            msgs.append(correctness_messages(q["question"], q["reference_answer"], a["text"]))
        for m in msgs:
            if not client.is_cached(m):
                calls += 1
                tokens += (
                    sum(len(x.content) for x in m) / config["generation_eval"]["chars_per_token"]
                )
    max_out = client.defaults["max_output_tokens"]
    print(
        f"{args.setup} ({args.split}): {len(todo)} answers to judge with {client.provider.name} "
        f"{client.model}; {calls} API calls, ~{tokens / 1000:.0f}K input tokens "
        f"+ up to {calls * max_out / 1000:.0f}K output"
    )
    if not args.run:
        print("Estimate only. Add --run to make the calls.")
        return

    try:
        for a in todo:
            q = questions[a["id"]]
            record = judge_answer(a, q, [by_id[i] for i in a["chunk_ids"]], client)
            append_record(out, {**record, "setup": args.setup})
            errors = [k for k in ("grounding", "correctness") if "error" in record.get(k, {})]
            print(f"  {a['id']}: done" + (f" (format error: {errors})" if errors else ""))
    except DailyQuotaReached as err:
        raise SystemExit(f"Stopped: {err}") from err
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

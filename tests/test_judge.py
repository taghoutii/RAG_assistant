import json

import pytest

from ragpapers.config import load_config
from ragpapers.evaluation import judge
from ragpapers.evaluation.judge import (
    JudgeFormatError,
    correctness_label,
    correctness_messages,
    grounding_labels,
    grounding_messages,
    judge_answer,
    load_human_grades,
    missing_grades,
    parse_json,
)
from ragpapers.generation.client import LLMClient
from ragpapers.generation.providers import FakeProvider

CONFIG = load_config()


def chunk(n):
    return {"chunk_id": f"p:{n}", "title": "Paper", "section": f"S{n}", "text": f"text {n}"}


CHUNKS = [chunk(n) for n in range(1, 9)]
QUESTION = {
    "id": "q101",
    "type": "factual",
    "question": "How often is the index refreshed?",
    "reference_answer": "About every 500 training steps.",
}
ANSWER = {"id": "q101", "text": "Every 500 steps [1].", "abstained": False, "invalid_citations": []}


def claim(supported=True, cited_support=True, citations=(1,)):
    return {
        "claim": "c",
        "citations": list(citations),
        "supported": supported,
        "cited_support": cited_support,
    }


# --- messages -----------------------------------------------------------------------------------


def test_grounding_call_sees_chunks_but_never_the_reference():
    system, user = grounding_messages(QUESTION["question"], CHUNKS, ANSWER["text"])
    assert "[8] Paper > S8\ntext 8" in user.content
    assert ANSWER["text"] in user.content and QUESTION["question"] in user.content
    assert QUESTION["reference_answer"] not in system.content + user.content


def test_correctness_call_sees_the_reference_but_not_the_chunks():
    system, user = correctness_messages(QUESTION["question"], QUESTION["reference_answer"], "A.")
    assert QUESTION["reference_answer"] in user.content
    assert "text 1" not in user.content


def test_braces_in_chunks_or_answers_survive():
    tricky = [chunk(1) | {"text": "x_{i} {answer}"}]
    _, user = grounding_messages("Q?", tricky, "uses {context} [1]")
    assert "x_{i} {answer}" in user.content and "uses {context} [1]" in user.content


# --- parsing ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"relevant": true, "claims": []}',
        '```json\n{"relevant": true, "claims": []}\n```',
        '<think>hmm {not json}</think>\n{"relevant": true, "claims": []}',
    ],
)
def test_parse_json_tolerates_fences_and_think_blocks(text):
    assert parse_json(text) == {"relevant": True, "claims": []}


@pytest.mark.parametrize("text", ["", "no json here", '{"claims": [}', '"just a string"'])
def test_parse_json_rejects_non_objects(text):
    with pytest.raises(JudgeFormatError):
        parse_json(text)


# --- labels from claims -------------------------------------------------------------------------


def test_all_claims_supported_and_cited():
    labels = grounding_labels({"claims": [claim(), claim()], "relevant": True}, [])
    assert labels == {
        "faithfulness": 1,
        "citation_correctness": 1,
        "relevance": 1,
        "n_claims": 2,
        "n_supported": 2,
    }


def test_one_unsupported_claim_fails_faithfulness():
    labels = grounding_labels({"claims": [claim(), claim(supported=False)], "relevant": True}, [])
    assert labels["faithfulness"] == 0 and labels["n_supported"] == 1


@pytest.mark.parametrize(
    ("claims", "invalid"),
    [
        ([claim(citations=())], []),  # uncited claim
        ([claim(cited_support=False)], []),  # cited passage does not support it
        ([claim()], [9]),  # cites a passage that was never shown
    ],
)
def test_citation_failures(claims, invalid):
    labels = grounding_labels({"claims": claims, "relevant": True}, invalid)
    assert labels["citation_correctness"] == 0 and labels["faithfulness"] == 1


def test_irrelevant_answer():
    assert grounding_labels({"claims": [claim()], "relevant": False}, [])["relevance"] == 0


@pytest.mark.parametrize(
    "parsed",
    [
        {"claims": [claim()]},  # no "relevant"
        {"claims": "none", "relevant": True},
        {
            "claims": [{"claim": "c", "citations": [1], "supported": "yes", "cited_support": True}],
            "relevant": True,
        },
    ],
)
def test_malformed_grounding_reply(parsed):
    with pytest.raises(JudgeFormatError):
        grounding_labels(parsed, [])


def test_correctness_needs_every_key_fact():
    facts = [{"fact": "500 steps", "present": True}, {"fact": "async", "present": False}]
    assert correctness_label({"key_facts": facts})["correctness"] == 0
    assert correctness_label({"key_facts": facts[:1]}) == {
        "correctness": 1,
        "n_key_facts": 1,
        "n_present": 1,
    }
    with pytest.raises(JudgeFormatError):
        correctness_label({"key_facts": []})


# --- judge_answer -------------------------------------------------------------------------------

GOOD_GROUNDING = json.dumps({"claims": [claim()], "relevant": True})
GOOD_CORRECTNESS = json.dumps({"key_facts": [{"fact": "500 steps", "present": True}]})


def test_answered_question_gets_two_calls_and_four_labels():
    fake = FakeProvider([GOOD_GROUNDING, GOOD_CORRECTNESS])
    record = judge_answer(ANSWER, QUESTION, CHUNKS, LLMClient(fake, "judge-m"))
    assert record["labels"] == {
        "faithfulness": 1,
        "citation_correctness": 1,
        "relevance": 1,
        "correctness": 1,
    }
    assert len(fake.requests) == 2
    assert record["grounding"]["model"] == "judge-m"
    assert QUESTION["reference_answer"] not in json.dumps(
        [m.content for m in fake.requests[0].messages]
    )


def test_abstention_makes_no_call_and_has_no_labels():
    fake = FakeProvider([])
    record = judge_answer(ANSWER | {"abstained": True}, QUESTION, CHUNKS, LLMClient(fake, "m"))
    assert record["labels"] == dict.fromkeys(judge.METRICS) and fake.requests == []


def test_answered_unanswerable_question_gets_no_correctness_call():
    fake = FakeProvider([GOOD_GROUNDING])
    q = QUESTION | {"type": "unanswerable", "reference_answer": CONFIG["abstention_text"]}
    record = judge_answer(ANSWER, q, CHUNKS, LLMClient(fake, "m"))
    assert record["labels"]["correctness"] is None and len(fake.requests) == 1


def test_format_error_is_recorded_not_raised():
    fake = FakeProvider(["I think it is fine.", GOOD_CORRECTNESS])
    record = judge_answer(ANSWER, QUESTION, CHUNKS, LLMClient(fake, "m"))
    assert "error" in record["grounding"] and record["grounding"]["raw"] == "I think it is fine."
    assert record["labels"]["faithfulness"] is None and record["labels"]["correctness"] == 1


def test_judge_requests_use_json_mode():
    fake = FakeProvider([GOOD_GROUNDING, GOOD_CORRECTNESS])
    judge_answer(ANSWER, QUESTION, CHUNKS, LLMClient(fake, "m", json_output=True))
    assert all(r.json_output for r in fake.requests)


# --- human grades and the blind guard -----------------------------------------------------------


def grade(qid, **labels):
    base = {
        "id": qid,
        "abstained": False,
        "faithfulness": 1,
        "citation_correctness": 1,
        "relevance": 1,
        "correctness": 1,
    }
    return base | labels


def test_missing_grades():
    grades = {
        "q1": grade("q1"),
        "q2": grade("q2", relevance=None),
        "q3": grade("q3", abstained=True, faithfulness=None),
    }
    assert missing_grades(grades, ["q1", "q2", "q3", "q4"]) == ["q2", "q4"]


def test_load_human_grades_is_strict(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(grade("q1")) + "\n" + json.dumps(grade("q2", relevance=2)) + "\n")
    with pytest.raises(ValueError, match="line 2"):
        load_human_grades(path)
    path.write_text(json.dumps(grade("q1")) + "\n{oops\n")
    with pytest.raises(ValueError, match="line 2"):
        load_human_grades(path)


def test_blind_guard_refuses_the_validation_setup_until_graded(tmp_path, monkeypatch):
    config = load_config()
    config["judge_validation"] = config["judge_validation"] | {
        "human_grades": str(tmp_path / "g.jsonl")
    }
    with pytest.raises(SystemExit, match="Blind grading"):
        judge.check_blind(config, "dev", "hybrid")
    judge.check_blind(config, "dev", "bm25")  # other setups are not the graded answers
    path = tmp_path / "g.jsonl"
    path.write_text("".join(json.dumps(grade(i)) + "\n" for i in judge.sample_ids(config)))
    judge.check_blind(config, "dev", "hybrid")  # complete grades: allowed


def test_sample_has_the_20_frozen_questions():
    ids = judge.sample_ids(CONFIG)
    assert len(ids) == 20 == len(set(ids)) and ids[0] == "q101"


def test_sample_only_judges_just_the_validation_answers(tmp_path, monkeypatch, capsys):
    answers = [{"id": i, "abstained": True} for i in ("q101", "q102", "q999")]
    monkeypatch.setattr(judge, "check_blind", lambda *a: None)
    monkeypatch.setattr(
        judge,
        "load_done",
        lambda path: {a["id"]: a for a in answers} if "answers" in str(path) else {},
    )
    monkeypatch.setattr(judge, "load_chunks", lambda path: [])
    monkeypatch.setattr(judge, "make_client", lambda config, role: LLMClient(FakeProvider([]), "m"))
    judge.main(["--setup", "hybrid", "--sample-only"])
    assert "2 answers to judge" in capsys.readouterr().out  # q999 is not in the sample


# --- format fix: bare key-facts list --------------------------------------------------------------

Q101_GEMINI = """[
  {"fact": "Rebuilds the index every several hundred steps.", "present": true},
  {"fact": "Documents are asynchronously re-embedded.", "present": true}
]"""


def test_bare_key_facts_list_is_accepted():
    """The exact shape of Gemini's q101 correctness reply."""
    assert correctness_label(parse_json(Q101_GEMINI)) == {
        "correctness": 1,
        "n_key_facts": 2,
        "n_present": 2,
    }
    fenced = "```json\n" + Q101_GEMINI + "\n```"
    assert correctness_label(parse_json(fenced))["correctness"] == 1


def test_bare_list_is_not_accepted_for_grounding_or_with_bad_items():
    with pytest.raises(JudgeFormatError):
        grounding_labels(parse_json(Q101_GEMINI), [])
    with pytest.raises(JudgeFormatError):
        correctness_label(parse_json("[1, 2]"))
    with pytest.raises(JudgeFormatError):
        correctness_label([])


def test_object_inside_prose_is_still_found():
    assert parse_json('Here you go: {"relevant": true, "claims": []} Done.') == {
        "relevant": True,
        "claims": [],
    }


def test_default_role_is_the_judge_kept_by_the_validation_rule(monkeypatch, capsys):
    roles = []
    monkeypatch.setattr(judge, "check_blind", lambda *a: None)
    monkeypatch.setattr(
        judge,
        "load_done",
        lambda path: {"q101": {"id": "q101", "abstained": True}} if "answers" in str(path) else {},
    )
    monkeypatch.setattr(judge, "load_chunks", lambda path: [])

    def fake_make_client(config, role):
        roles.append(role)
        return LLMClient(FakeProvider([]), "m")

    monkeypatch.setattr(judge, "make_client", fake_make_client)
    judge.main(["--setup", "hybrid"])
    judge.main(["--setup", "hybrid", "--role", "judge"])
    assert roles == [CONFIG["judge"]["kept"], "judge"]
    assert CONFIG["judge"]["kept"] == "judge_fallback"

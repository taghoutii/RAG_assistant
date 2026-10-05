import json

import pytest

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.questions import (
    check_quotes,
    check_schema,
    check_set,
    load_questions,
    questions_per_paper,
    validate,
)
from ragpapers.ingest.manifest import load_manifest

CONFIG = load_config()
CFG = CONFIG["eval"]
ABSTAIN = CONFIG["abstention_text"]
PAPER_IDS = {"1111.00001", "2222.00002"}
CORPUS = {
    "1111.00001": {
        "sections": [
            {
                "section_id": "S1",
                "path": ["1 Intro"],
                "blocks": [{"text": "The model uses eight attention heads in every layer."}],
            }
        ]
    },
    "2222.00002": {
        "sections": [
            {
                "section_id": "S3",
                "path": ["3 Setup"],
                "blocks": [{"text": "We train with a batch size of 512 sequences for 100k steps."}],
            }
        ]
    },
}


def factual(**overrides):
    q = {
        "id": "q001",
        "question": "How many attention heads does the model use?",
        "type": "factual",
        "reference_answer": "Eight.",
        "evidence": [{"paper_id": "1111.00001", "quote": "uses eight attention heads in every"}],
    }
    q.update(overrides)
    return q


def comparison(**overrides):
    q = {
        "id": "q002",
        "question": "Compare heads in paper one with the batch size in paper two?",
        "type": "comparison",
        "reference_answer": "Eight heads; batch size 512.",
        "evidence": [
            {"paper_id": "1111.00001", "quote": "uses eight attention heads in every"},
            {"paper_id": "2222.00002", "quote": "a batch size of 512 sequences"},
        ],
    }
    q.update(overrides)
    return q


def unanswerable(**overrides):
    q = {
        "id": "q003",
        "question": "What dropout rate was used?",
        "type": "unanswerable",
        "reference_answer": ABSTAIN,
        "evidence": [],
    }
    q.update(overrides)
    return q


def errors(issues):
    return [i.message for i in issues if i.level == "error"]


def warnings(issues):
    return [i.message for i in issues if i.level == "warning"]


@pytest.mark.parametrize("q", [factual(), comparison(), unanswerable()])
def test_valid_questions_have_no_issues(q):
    issues = check_schema(q, PAPER_IDS, CFG, ABSTAIN) + check_quotes(q, CORPUS, 90)
    assert issues == []


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ({k: v for k, v in factual().items() if k != "evidence"}, "missing fields"),
        (factual(answer="typo field"), "unknown fields"),
        (factual(id="Q-1"), "must look like q001"),
        (factual(type="opinion"), "type 'opinion'"),
        (factual(reference_answer=" "), "reference_answer must be"),
        (factual(evidence=[]), "at least one evidence"),
        (factual(evidence=comparison()["evidence"]), "use type 'comparison'"),
        (comparison(evidence=factual()["evidence"]), "at least 2 different papers"),
        (unanswerable(evidence=factual()["evidence"]), "must have empty evidence"),
        (unanswerable(reference_answer="No idea."), "must be exactly"),
        (factual(evidence=[{"paper_id": "9999.99999", "quote": "a b c d e"}]), "not in the"),
        (factual(evidence=[{"paper": "1111.00001", "quote": "a b c d e"}]), "exactly 'paper_id'"),
    ],
)
def test_schema_errors(q, expected):
    messages = errors(check_schema(q, PAPER_IDS, CFG, ABSTAIN))
    assert any(expected in m for m in messages), messages


def test_schema_warnings_for_quote_length_and_question_mark():
    short = factual(question="Heads", evidence=[{"paper_id": "1111.00001", "quote": "eight"}])
    messages = warnings(check_schema(short, PAPER_IDS, CFG, ABSTAIN))
    assert any("does not end with '?'" in m for m in messages)
    assert any("quote has 1 words" in m for m in messages)


def test_quote_with_typo_is_a_warning_and_missing_quote_an_error():
    typo = factual(evidence=[{"paper_id": "1111.00001", "quote": "uses eight atention heads in"}])
    assert warnings(check_quotes(typo, CORPUS, 90))
    missing = factual(evidence=[{"paper_id": "1111.00001", "quote": "we use rotary embeddings"}])
    assert errors(check_quotes(missing, CORPUS, 90))


def test_quote_from_wrong_paper_suggests_the_right_one():
    wrong = factual(evidence=[{"paper_id": "1111.00001", "quote": "a batch size of 512 sequences"}])
    (message,) = errors(check_quotes(wrong, CORPUS, 90))
    assert "It does appear in paper 2222.00002" in message


def test_set_checks_duplicates_and_mix():
    questions = [factual(), factual(question="How many attention heads does the model use ?")]
    messages = [i.message for i in check_set(questions, CFG)]
    assert any("id used 2 times" in m for m in messages)
    assert any("near-duplicate of q001" in m for m in messages)
    assert any("unanswerable: 0%" in m for m in messages)


def test_questions_per_paper_counts_each_cited_paper_once():
    questions = [factual(), comparison(), unanswerable()]
    counts = questions_per_paper(questions, ["1111.00001", "2222.00002", "3333.00003"])
    assert counts == {"1111.00001": 2, "2222.00002": 1, "3333.00003": 0}


def test_load_questions_reports_line_number(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(factual()) + "\n\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        load_questions(path)


@pytest.mark.parametrize("name", ["questions.jsonl", "candidates.jsonl"])
def test_committed_eval_files_have_no_schema_errors(name):
    """Runs in CI too: schema only, because the parsed corpus is not committed."""
    path = REPO_ROOT / "eval" / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    paper_ids = {p.arxiv_id for p in load_manifest(REPO_ROOT / "data" / "corpus_manifest.yaml")}
    issues = validate(load_questions(path), paper_ids, CFG, ABSTAIN, corpus=None)
    assert errors(issues) == []

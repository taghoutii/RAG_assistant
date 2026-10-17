import json

from ragpapers.evaluation.alternates import (
    apply_ticked,
    build_candidates,
    candidate_sentences,
    read_ticked,
    split_sentences,
    write_questions,
)

CFG = {
    "min_shared_words": 4,
    "min_shared_fraction": 0.2,
    "same_text_threshold": 90,
    "quote_min_words": 5,
    "quote_max_words": 50,
    "alternatives_max": 3,
}
QUOTE = "we refresh the document index every five hundred training steps"
RESTATED = "The retriever index is rebuilt after every five hundred training steps of the model."
PAPER = {
    "title": "A Paper",
    "sections": [
        {
            "section_id": "S0",
            "path": ["Abstract"],
            "blocks": [{"type": "paragraph", "text": f"Intro words here. {RESTATED}"}],
        },
        {
            "section_id": "S3",
            "path": ["3 Method"],
            "blocks": [
                {"type": "paragraph", "text": f"In practice {QUOTE}. Nothing else matters."},
                {"type": "table", "text": "index rebuilt every five hundred training steps"},
            ],
        },
    ],
}
ANSWER = "The document index is rebuilt every five hundred training steps."
QUESTION = {
    "id": "q001",
    "question": "How often is the index rebuilt?",
    "type": "factual",
    "reference_answer": ANSWER,
    "evidence": [{"paper_id": "p1", "quote": QUOTE}],
}


def test_split_sentences():
    assert split_sentences("One two. Three four? (Five) six!  Seven") == [
        "One two.",
        "Three four?",
        "(Five) six!",
        "Seven",
    ]


def test_candidates_find_the_restatement_and_skip_the_quote_and_tables():
    found = candidate_sentences(PAPER, ANSWER, QUOTE, CFG)
    assert [c["text"] for c in found] == [RESTATED]
    assert found[0]["section"] == "Abstract"


def test_candidates_need_enough_shared_words():
    assert candidate_sentences(PAPER, ANSWER, QUOTE, {**CFG, "min_shared_words": 9}) == []


def test_candidates_need_enough_of_the_answer():
    # A long answer makes 20% of its content words more than this sentence shares.
    long_answer = ANSWER + " " + " ".join(f"word{i}" for i in range(40))
    assert candidate_sentences(PAPER, long_answer, QUOTE, CFG) == []


def test_build_candidates_ids_and_fields():
    candidates = build_candidates([QUESTION], {"p1": PAPER}, CFG)
    assert list(candidates) == ["q001.e0.c0"]
    assert candidates["q001.e0.c0"]["paper_id"] == "p1"
    assert "alternatives" not in candidates["q001.e0.c0"]


BATCH = """## q001

- [x] `q001.e0.c0` (14 words) Abstract
  > The retriever index is rebuilt after every five hundred training steps
- [ ] `q001.e0.c1` (9 words) Other
  > not ticked
"""


def test_read_ticked_takes_the_possibly_trimmed_text():
    assert read_ticked(BATCH) == {
        "q001.e0.c0": "The retriever index is rebuilt after every five hundred training steps"
    }


def fresh():
    return [json.loads(json.dumps(QUESTION))]


def test_apply_adds_alternatives_and_is_idempotent():
    questions, corpus = fresh(), {"p1": PAPER}
    candidates = build_candidates(questions, corpus, CFG)
    ticked = read_ticked(BATCH)
    assert apply_ticked(questions, ticked, candidates, corpus, CFG) == []
    assert apply_ticked(questions, ticked, candidates, corpus, CFG) == []
    assert questions[0]["evidence"][0]["alternatives"] == [ticked["q001.e0.c0"]]


def test_apply_rejects_edited_text_and_changes_nothing():
    questions, corpus = fresh(), {"p1": PAPER}
    candidates = build_candidates(questions, corpus, CFG)
    problems = apply_ticked(
        questions,
        {"q001.e0.c0": "The index is rebuilt every 500 steps, says the paper."},
        candidates,
        corpus,
        CFG,
    )
    assert any("not verbatim" in p for p in problems)
    assert "alternatives" not in questions[0]["evidence"][0]


def test_apply_rejects_too_short_and_unknown_ids():
    questions, corpus = fresh(), {"p1": PAPER}
    candidates = build_candidates(questions, corpus, CFG)
    problems = apply_ticked(
        questions,
        {"q001.e0.c0": "five hundred training steps", "q009.e0.c0": "x"},
        candidates,
        corpus,
        CFG,
    )
    assert any("words" in p for p in problems)
    assert any("unknown" in p for p in problems)


def test_apply_enforces_the_maximum():
    questions, corpus = fresh(), {"p1": PAPER}
    questions[0]["evidence"][0]["alternatives"] = ["a b c d e", "f g h i j", "k l m n o"]
    candidates = build_candidates(questions, corpus, CFG)
    problems = apply_ticked(questions, read_ticked(BATCH), candidates, corpus, CFG)
    assert any("max 3" in p for p in problems)


def test_write_questions_keeps_the_hand_written_format(tmp_path):
    path = tmp_path / "q.jsonl"
    write_questions(path, [QUESTION, {**QUESTION, "id": "q002", "notes": "é"}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == json.dumps(QUESTION, ensure_ascii=False)
    assert "é" in lines[1]

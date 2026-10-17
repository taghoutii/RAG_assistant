from ragpapers.evaluation.rescore_phase4 import rescore, stored_rankings

CFG = {"top_n": 2, "ks": [1, 2], "quote_match_threshold": 90}
QUOTE = "the decoder attends over all encoded passages"
ALT = "evidence fusion happens in the decoder only"
BY_ID = {
    "a": {"chunk_id": "a", "text": f"We note that {ALT}."},
    "b": {"chunk_id": "b", "text": f"In FiD {QUOTE} jointly."},
    "c": {"chunk_id": "c", "text": "unrelated"},
}
QUESTIONS = {
    "q1": {"id": "q1", "evidence": [{"paper_id": "p", "quote": QUOTE, "alternatives": [ALT]}]}
}
ENTRIES = [{"id": "q1", "type": "factual", "retrieved": ["a", "b", "c"]}]


def test_original_ignores_alternatives():
    (row,) = rescore(ENTRIES, QUESTIONS, BY_ID, CFG, alternatives=False)
    assert row["scores"]["rr"] == 0.5  # the quote is only at rank 2


def test_revised_credits_the_alternative_first():
    (row,) = rescore(ENTRIES, QUESTIONS, BY_ID, CFG, alternatives=True)
    assert row["scores"]["rr"] == 1.0
    assert row["scores"]["recall@1"] == 1.0  # one evidence item, found once


def test_only_the_top_n_is_scored():
    entries = [{"id": "q1", "type": "factual", "retrieved": ["c", "c2", "b"]}]
    by_id = {**BY_ID, "c2": {"chunk_id": "c2", "text": "also unrelated"}}
    (row,) = rescore(entries, QUESTIONS, by_id, CFG, alternatives=True)
    assert row["scores"]["recall@2"] == 0.0


def test_stored_rankings_flat_and_keyed():
    flat = {"per_question": ENTRIES, "metrics": {"m": 1}}
    keyed = {"per_question": {"20": ENTRIES}, "by_depth": {"20": {"metrics": {"m": 2}}}}
    assert stored_rankings(flat, None) == (ENTRIES, {"m": 1})
    assert stored_rankings(keyed, "20") == (ENTRIES, {"m": 2})

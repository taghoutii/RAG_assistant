from ragpapers.evaluation.run_phase4 import STAGES, evidence_ranks

Q1 = "the decoder attends over all encoded passages"
Q2 = "retrieval is performed once per generated chunk"


def test_evidence_ranks_first_occurrence_per_quote():
    texts = ["unrelated", f"In FiD {Q1} jointly.", f"Again: {Q1}.", f"In RETRO {Q2} here."]
    assert evidence_ranks(texts, [[Q1], [Q2]], 90) == [2, 4]


def test_evidence_ranks_missing_quote_is_none():
    assert evidence_ranks(["unrelated"], [[Q1]], 90) == [None]


def test_hybrid_rerank_runs_every_stage():
    assert STAGES["hybrid+rerank"] == ("bm25", "dense", "fusion", "rerank")
    assert set(STAGES["hybrid"]) < set(STAGES["hybrid+rerank"])

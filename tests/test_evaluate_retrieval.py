from ragpapers.evaluation.evaluate_retrieval import METRICS, SETUPS, STAGES, make_report


def test_hybrid_rerank_runs_every_stage():
    assert STAGES["hybrid+rerank"] == ("bm25", "dense", "fusion", "rerank")
    assert set(STAGES["hybrid"]) < set(STAGES["hybrid+rerank"])


def fake_result(name, value):
    keys = ["rr" if m == "mrr" else m for m in METRICS]  # per-question scores name MRR "rr"
    per_question = [{"id": f"q{i}", "scores": dict.fromkeys(keys, value)} for i in range(3)]
    metric = {"mean": value, "ci_low": value, "ci_high": value}
    return {"name": name, "metrics": dict.fromkeys(METRICS, metric), "per_question": per_question}


def test_report_lists_every_setup_and_the_paired_comparisons():
    results = {name: fake_result(name, 0.5) for name in SETUPS}
    latency = {
        "stages": {"bm25": 7.5, "dense": 25.4, "fusion": 0.1, "rerank": 1200.0},
        "total": dict.fromkeys(SETUPS, 33.0),
    }
    meta = {
        "bootstrap_samples": 10,
        "seed": 0,
        "chunk_config": "tokens448_ov64",
        "n_chunks": 3,
        "rrf_k": 60,
        "fusion_depth": 100,
        "reranker": "m",
        "rerank_depth": 20,
        "top_n": 10,
        "n_answerable": 3,
        "n_latency_questions": 4,
    }
    report = make_report(results, latency, meta)
    for name in SETUPS:
        assert f"| {name} | 0.500 [0.50, 0.50]" in report
    assert "## Paired differences against hybrid" in report
    assert "| bm25 | 7.5 | - | - | - | 33 |" in report

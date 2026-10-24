from ragpapers.config import load_config
from ragpapers.evaluation.evaluate_generation import (
    fmt_metric,
    make_report,
    score_questions,
    summarize,
    usage,
)


def answer(qid, qtype, abstained, latency=500.0):
    return {
        "id": qid,
        "type": qtype,
        "abstained": abstained,
        "input_tokens": 3000,
        "output_tokens": 100,
        "reasoning_tokens": 20,
        "latency_ms": {"bm25": 5.0, "generation": latency},
    }


def labels(f=1, c=1, r=1, k=1):
    return {
        "labels": {"faithfulness": f, "citation_correctness": c, "relevance": r, "correctness": k}
    }


ANSWERS = {
    "q1": answer("q1", "factual", False),
    "q2": answer("q2", "comparison", True),  # false abstention
    "q3": answer("q3", "unanswerable", True),  # correct abstention
    "q4": answer("q4", "unanswerable", False),  # missed abstention
}
JUDGMENTS = {"q1": labels(c=0, k=0), "q4": labels(f=0, k=None)}


def scores(per_question):
    return {q["id"]: q["scores"] for q in per_question}


def test_which_metrics_apply_to_which_questions():
    s = scores(score_questions(ANSWERS, JUDGMENTS))
    assert [s[q]["abstention_accuracy"] for q in ("q1", "q2", "q3", "q4")] == [1, 0, 1, 0]
    assert s["q1"] == {
        "abstention_accuracy": 1,
        "faithfulness": 1,
        "citation_correctness": 0,
        "relevance": 1,
        "correctness": 0,
    }
    # abstentions: no judge metric at all
    assert all(
        s[q][m] is None
        for q in ("q2", "q3")
        for m in ("faithfulness", "citation_correctness", "relevance", "correctness")
    )
    # answered unanswerable: grounding labels, no correctness
    assert s["q4"]["faithfulness"] == 0 and s["q4"]["correctness"] is None


def test_summary_means_ns_and_missing_judgments():
    per_question = score_questions(ANSWERS, {"q1": labels()})  # q4 not judged yet
    m = summarize(per_question, 200, 42)
    assert m["abstention_accuracy"]["mean"] == 0.5 and m["abstention_accuracy"]["n"] == 4
    assert m["faithfulness"]["n"] == 1 and m["faithfulness"]["n_missing"] == 1
    assert m["correctness"]["n"] == 1 and m["correctness"]["n_missing"] == 0
    low, high = m["abstention_accuracy"]["ci_low"], m["abstention_accuracy"]["ci_high"]
    assert 0 <= low <= 0.5 <= high <= 1


def test_unjudged_setup_says_so():
    m = summarize(score_questions(ANSWERS, {}), 100, 42)
    assert fmt_metric(m["faithfulness"]) == "not judged"
    assert fmt_metric(m["abstention_accuracy"]).startswith("0.500 [")


def test_usage_marks_unmeasured_latency():
    u = usage({"q1": answer("q1", "factual", False, latency=None)})
    assert u["generation_latency_ms_p50"] is None and u["generation_latency_n"] == 0
    assert usage(ANSWERS)["generation_latency_ms_p50"] == 500.0
    assert usage(ANSWERS)["cost_usd"] == 0.0


def test_report_puts_agreement_next_to_each_judge_metric():
    agreement = {
        m: {"kappa": 0.5, "raw": 0.8, "n": 19}
        for m in ("faithfulness", "citation_correctness", "relevance", "correctness")
    }
    agreement["relevance"]["kappa"] = None
    per_question = score_questions(ANSWERS, JUDGMENTS)
    result = {"metrics": summarize(per_question, 100, 42), "usage": usage(ANSWERS)}
    meta = {
        "split": "dev",
        "generator": "g",
        "context_k": 8,
        "judge": "j",
        "n_graded": 19,
        "bootstrap_samples": 100,
        "seed": 42,
    }
    report = make_report({"hybrid": result}, agreement, meta)
    row = next(line for line in report.splitlines() if line.startswith("| Judge agreement"))
    assert row.count("kappa") == 4 and "kappa n/a" in row
    assert "(reliable)" in row and "(indicative)" in row
    assert "| Hybrid (RRF) |" in report and "500 ms" in report


def test_the_judge_is_qwen_with_no_fallback_left_in_the_config():
    judge = load_config()["judge"]
    assert (judge["provider"], judge["model"]) == ("groq", "qwen/qwen3.8-27b")
    assert not any(key.startswith("fallback") or key == "kept" for key in judge)


def test_each_cell_carries_its_own_n():
    m = summarize(score_questions(ANSWERS, JUDGMENTS), 100, 42)
    assert fmt_metric(m["faithfulness"]).endswith("n=2")
    assert fmt_metric(m["abstention_accuracy"]).endswith("n=4")

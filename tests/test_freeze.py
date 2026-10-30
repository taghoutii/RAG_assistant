import pytest

from ragpapers.evaluation import evaluate_generation, evaluate_retrieval, freeze, judge
from ragpapers.evaluation.evaluate_generation import check_same_commit
from ragpapers.evaluation.evaluate_retrieval import make_report

COMMIT = "b33b92b0000000000000000000000000000000000"


def fake_git(changes=""):
    calls = []

    def git(*args):
        calls.append(args)
        return changes if args[0] == "status" else COMMIT

    return git, calls


def test_frozen_commit_when_the_frozen_paths_are_clean(monkeypatch):
    git, calls = fake_git()
    monkeypatch.setattr(freeze, "_git", git)
    assert freeze.frozen_commit() == COMMIT
    status = calls[0]
    assert "--untracked-files=no" in status and set(freeze.FROZEN_PATHS) <= set(status)


def test_uncommitted_change_to_a_frozen_file_is_refused(monkeypatch):
    monkeypatch.setattr(freeze, "_git", fake_git(" M src/ragpapers/generation/prompt.py")[0])
    with pytest.raises(SystemExit, match="prompt.py"):
        freeze.frozen_commit()


def test_require_final_flag(monkeypatch):
    git, calls = fake_git()
    monkeypatch.setattr(freeze, "_git", git)
    assert freeze.require_final_flag("dev", False) is None and calls == []  # dev: no git at all
    with pytest.raises(SystemExit, match="held out"):
        freeze.require_final_flag("test", False)
    assert freeze.require_final_flag("test", True) == COMMIT


@pytest.mark.parametrize("main", [evaluate_retrieval.main, judge.main, evaluate_generation.main])
def test_every_test_step_needs_the_final_flag(main):
    argv = ["--split", "test"] + (["--setup", "hybrid"] if main is judge.main else [])
    with pytest.raises(SystemExit, match="held out"):
        main(argv)


def test_test_records_must_all_come_from_the_frozen_commit():
    same = {"q1": {"commit": COMMIT}, "q2": {"commit": COMMIT}}
    check_same_commit(COMMIT, same, {"q1": {"commit": COMMIT}})
    with pytest.raises(SystemExit, match="not b33b92b"):
        check_same_commit(COMMIT, same, {"q1": {"commit": "other"}})
    with pytest.raises(SystemExit):
        check_same_commit(COMMIT, {"q1": {}})  # no commit recorded


def retrieval_inputs():
    metric = {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}
    names = ("recall@3", "recall@5", "recall@10", "mrr", "ndcg@10")
    scores = dict.fromkeys(("recall@3", "recall@5", "recall@10", "rr", "ndcg@10"), 0.5)
    results = {
        s: {
            "name": s,
            "metrics": dict.fromkeys(names, metric),
            "per_question": [{"id": "q1", "scores": scores}],
        }
        for s in evaluate_retrieval.SETUPS
    }
    latency = {
        "stages": dict.fromkeys(("bm25", "dense", "fusion", "rerank"), 1.0),
        "total": dict.fromkeys(evaluate_retrieval.SETUPS, 2.0),
    }
    meta = {
        "bootstrap_samples": 10,
        "seed": 42,
        "chunk_config": "c",
        "n_chunks": 1,
        "rrf_k": 60,
        "fusion_depth": 100,
        "reranker": "r",
        "rerank_depth": 20,
        "top_n": 10,
        "n_answerable": 29,
        "n_latency_questions": 36,
    }
    return results, latency, meta


def test_retrieval_report_names_its_split():
    results, latency, meta = retrieval_inputs()
    dev = make_report(results, latency, meta)  # no split given: dev, as before
    assert dev.startswith("# Retrieval on dev") and "answerable dev questions" in dev
    test = make_report(results, latency, meta | {"split": "test"})
    assert test.startswith("# Retrieval on test") and "answerable test questions" in test
    assert "dev" not in test.split("\n", 1)[0] and "36 test questions" in test

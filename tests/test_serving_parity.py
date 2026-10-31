"""Parity: the served system reproduces the evaluated one, on every hybrid dev answer.

For each dev question of reports/results/generation/dev/hybrid.answers.jsonl, the service
must retrieve the same 8 chunk IDs and send the generator the same request (same request
hash), so the answer is served from the LLM cache and is identical to the evaluated one.
It cannot make an API call: the Groq provider is replaced by one that fails if called.

Needs the local corpus, indexes and LLM cache (gitignored), so it is opt-in and CI skips it.
Stop the API first (the local Qdrant index allows one process), then run:
    pytest -m parity
"""

import pytest

from ragpapers.api.querylog import QueryLog, read_log
from ragpapers.api.service import RagService
from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.questions import load_questions
from ragpapers.generation import client as client_module
from ragpapers.generation.client import make_client
from ragpapers.generation.providers import Provider
from ragpapers.generation.runlog import load_done

pytestmark = pytest.mark.parity

CONFIG = load_config()
ANSWERS = REPO_ROOT / CONFIG["paths"]["results_dir"] / "generation" / "dev" / "hybrid.answers.jsonl"
NEEDED = [
    ANSWERS,
    REPO_ROOT / CONFIG["paths"]["chunks_dir"],
    REPO_ROOT / CONFIG["index"]["qdrant_path"],
    REPO_ROOT / CONFIG["llm_calls"]["cache_dir"],
]


class NoCallGroq(Provider):
    name = "groq"  # same cache folder and request hash as the real provider

    def send(self, request):
        raise AssertionError("the parity check tried to call the API: the request changed")


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    missing = [str(p.relative_to(REPO_ROOT)) for p in NEEDED if not p.exists()]
    if missing:
        pytest.skip(f"local data missing: {', '.join(missing)}")
    mp = pytest.MonkeyPatch()
    mp.setenv("GROQ_API_KEY", "unused-by-the-parity-test")
    mp.setattr(client_module, "build_provider", lambda *args: NoCallGroq())
    from ragpapers.retrieval.pipeline import load_indexes

    log = QueryLog(tmp_path_factory.mktemp("parity") / "queries.jsonl")
    svc = RagService(CONFIG, load_indexes(CONFIG), make_client(CONFIG, "generator"), log)
    yield svc
    svc.close()
    mp.undo()


def test_every_hybrid_dev_answer_is_reproduced_from_the_cache(service):
    evaluated = load_done(ANSWERS)
    questions = {
        q["id"]: q["question"]
        for q in load_questions(REPO_ROOT / CONFIG["paths"]["eval_questions"])
    }
    assert len(evaluated) > 30

    for qid, record in evaluated.items():
        result = service.ask(questions[qid])
        assert [c["chunk_id"] for c in result["retrieved"]] == record["chunk_ids"], qid
        assert result["cached"] is True, qid
        assert result["abstained"] == record["abstained"], qid
        assert [c["number"] for c in result["citations"]] == record["cited"], qid

    logged = read_log(service.log.path)
    assert [r["request_hash"] for r in logged] == [r["request_hash"] for r in evaluated.values()]

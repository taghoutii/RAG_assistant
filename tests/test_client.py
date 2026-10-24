import json
import os

import pytest

from ragpapers.config import load_config
from ragpapers.generation.client import (
    DailyQuotaReached,
    LLMClient,
    MinuteThrottle,
    make_client,
    request_hash,
    safe_name,
)
from ragpapers.generation.env import load_env
from ragpapers.generation.providers import (
    FakeProvider,
    GroqProvider,
    LLMRequest,
    Message,
    OllamaProvider,
    ProviderError,
    RateLimited,
    TransientError,
)
from ragpapers.generation.runlog import append_record, load_done

PROMPT = [Message("user", "What is RRF?")]


class FakeClock:
    """Time that only moves when the code under test sleeps."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def client(script, tmp_path=None, limits=None, **kwargs):
    clock = FakeClock()
    fake = FakeProvider(script)
    c = LLMClient(
        fake,
        "model-a",
        limits=limits,
        cache_dir=tmp_path,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )
    return c, fake, clock


# --- cache --------------------------------------------------------------------------------------


def test_second_identical_call_comes_from_the_cache(tmp_path):
    c, fake, _ = client(["answer"], tmp_path)
    first = c.generate(PROMPT)
    second = c.generate(PROMPT)
    assert len(fake.requests) == 1  # no second API call
    assert (first.cached, second.cached) == (False, True)
    assert second.text == "answer" and second.request_hash == first.request_hash
    assert second.latency_ms == first.latency_ms  # the original call's latency is kept


def test_cache_file_holds_request_and_response(tmp_path):
    c, _, _ = client(["answer"], tmp_path, temperature=0.0, max_output_tokens=64)
    r = c.generate(PROMPT)
    (path,) = (tmp_path / "fake" / "model-a").glob("*.json")
    assert path.stem == r.request_hash
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["request"]["messages"] == [{"role": "user", "content": "What is RRF?"}]
    assert record["request"]["max_output_tokens"] == 64
    assert record["response"]["text"] == "answer"
    assert record["response"]["requested_model"] == "model-a"
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_cached_answer_survives_a_new_client(tmp_path):
    client(["answer"], tmp_path)[0].generate(PROMPT)
    c, fake, _ = client([], tmp_path)  # empty script: any API call would fail
    assert c.generate(PROMPT).text == "answer" and fake.requests == []


@pytest.mark.parametrize(
    ("messages", "params"),
    [
        ([Message("user", "What is BM25?")], {}),
        ([Message("system", "Be brief."), *PROMPT], {}),
        (PROMPT, {"temperature": 0.7}),
        (PROMPT, {"max_output_tokens": 10}),
        (PROMPT, {"reasoning_effort": "high"}),
    ],
)
def test_any_change_to_prompt_or_settings_misses_the_cache(tmp_path, messages, params):
    c, fake, _ = client(["first", "second"], tmp_path)
    c.generate(PROMPT)
    assert c.generate(messages, **params).text == "second"
    assert len(fake.requests) == 2


def test_hash_depends_on_provider_and_model():
    req = LLMRequest("m", PROMPT, 0.0, 10)
    assert request_hash("groq", req) != request_hash("gemini", req)
    assert request_hash("groq", req) != request_hash("groq", LLMRequest("m2", PROMPT, 0.0, 10))
    assert request_hash("groq", req) == request_hash("groq", LLMRequest("m", PROMPT, 0.0, 10))


def test_unknown_generation_parameter_is_rejected():
    c, _, _ = client(["x"])
    with pytest.raises(TypeError):
        c.generate(PROMPT, top_p=0.9)


def test_safe_name():
    assert safe_name("openai/gpt-oss-120b") == "openai_gpt-oss-120b"
    assert safe_name("qwen3:8b") == "qwen3_8b"


def test_response_records_model_ids_and_metadata():
    fake = FakeProvider(["ok"], model_reported="model-a-2026-10")
    r = LLMClient(fake, "model-a").generate(PROMPT)
    assert (r.requested_model, r.model) == ("model-a", "model-a-2026-10")
    assert r.request_hash and r.created_at and r.latency_ms is not None and not r.cached


# --- retries ------------------------------------------------------------------------------------


def test_429_is_retried_and_retry_after_is_honoured():
    c, fake, clock = client([RateLimited("slow", retry_after=7), "ok"])
    assert c.generate(PROMPT).text == "ok"
    assert len(fake.requests) == 2
    assert 7 <= clock.sleeps[0] <= 8  # Retry-After + up to 1 s jitter


def test_backoff_doubles_with_jitter_and_is_capped():
    errors = [TransientError("503")] * 5
    c, _, clock = client([*errors, "ok"], backoff_initial_s=2, backoff_max_s=10, max_retries=6)
    c.generate(PROMPT)
    # 2, 4, 8, 10, 10 seconds before jitter; jitter keeps 50-100% of each, plus up to 1 s
    for sleep, base in zip(clock.sleeps, [2, 4, 8, 10, 10], strict=True):
        assert 0.5 * base <= sleep <= base + 1


def test_gives_up_after_max_retries():
    c, fake, _ = client([TransientError("503")] * 4, max_retries=3)
    with pytest.raises(TransientError):
        c.generate(PROMPT)
    assert len(fake.requests) == 4  # first try + 3 retries


def test_other_errors_fail_fast(tmp_path):
    c, fake, clock = client([ProviderError("HTTP 401: Invalid API Key")], tmp_path)
    with pytest.raises(ProviderError):
        c.generate(PROMPT)
    assert len(fake.requests) == 1 and clock.sleeps == []
    assert not list(tmp_path.rglob("*.json"))  # failures are never cached


@pytest.mark.parametrize(
    "error",
    [RateLimited("tokens per day (TPD)", daily=True), RateLimited("wait", retry_after=3600)],
)
def test_daily_quota_stops_instead_of_sleeping(error):
    c, fake, clock = client([error, "never reached"], backoff_max_s=120)
    with pytest.raises(DailyQuotaReached, match="re-run"):
        c.generate(PROMPT)
    assert len(fake.requests) == 1 and clock.sleeps == []


# --- per-minute throttle ------------------------------------------------------------------------


def test_throttle_waits_for_the_rpm_window():
    clock = FakeClock()
    throttle = MinuteThrottle(rpm=2, tpm=None, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        throttle.wait(10)
        throttle.record(10)
    assert len(clock.sleeps) == 1 and clock.now >= 60


def test_throttle_waits_for_the_tpm_window():
    clock = FakeClock()
    throttle = MinuteThrottle(rpm=None, tpm=8000, clock=clock, sleep=clock.sleep)
    throttle.wait(5000)
    throttle.record(5000)
    clock.now = 20
    throttle.wait(5000)  # 5000 + 5000 > 8000: waits until the first call leaves the window
    assert clock.now >= 60


def test_throttle_always_lets_a_single_call_through():
    clock = FakeClock()
    MinuteThrottle(rpm=1, tpm=100, clock=clock, sleep=clock.sleep).wait(10_000)
    assert clock.sleeps == []


def test_client_spaces_calls_under_the_rpm_limit():
    c, fake, clock = client(["a", "b", "c"], limits={"rpm": 2, "tpm": None})
    for i in range(3):
        c.generate([Message("user", f"question {i}")])
    assert len(fake.requests) == 3 and clock.now >= 60


# --- resumable runs -----------------------------------------------------------------------------


def test_run_log_resumes_and_ignores_a_cut_off_last_line(tmp_path):
    path = tmp_path / "run.jsonl"
    append_record(path, {"id": "q101", "answer": "a"})
    append_record(path, {"id": "q102", "answer": "b"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"id": "q103", "ans')  # crash mid-write
    assert set(load_done(path)) == {"q101", "q102"}
    assert load_done(tmp_path / "missing.jsonl") == {}
    with pytest.raises(ValueError):
        append_record(path, {"answer": "no id"})


# --- .env and make_client -----------------------------------------------------------------------


def test_load_env_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nNEW_KEY="abc"\nexport OTHER=x\nKEEP=from-file\nnot a line\n')
    monkeypatch.delenv("NEW_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("KEEP", "from-shell")
    load_env(env)
    assert os.environ["NEW_KEY"] == "abc" and os.environ["OTHER"] == "x"
    assert os.environ["KEEP"] == "from-shell"
    load_env(tmp_path / "missing.env")  # no file: nothing happens


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")


def test_make_client_follows_the_config(tmp_path, keys):
    config = load_config()
    no_env = tmp_path / "none.env"

    gen = make_client(config, "generator", env_file=no_env)
    assert isinstance(gen.provider, GroqProvider) and gen.model == config["llm"]["model"]
    assert gen.defaults == {
        "temperature": 0,
        "max_output_tokens": config["llm"]["max_output_tokens"],
        "reasoning_effort": "low",
        "json_output": False,
    }
    assert gen.throttle.rpm == config["llm"]["limits"]["rpm"]

    judge = make_client(config, "judge", env_file=no_env)
    assert isinstance(judge.provider, GroqProvider)
    assert judge.model == config["judge"]["model"] == "qwen/qwen3.8-27b"
    assert judge.defaults["reasoning_effort"] is None
    assert judge.throttle.tpm == config["judge"]["limits"]["tpm"]

    with pytest.raises(ValueError, match="unknown role"):
        make_client(config, "judge_fallback", env_file=no_env)


def test_make_client_explains_a_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        make_client(load_config(), "generator", env_file=tmp_path / "none.env")


def test_make_client_builds_ollama_without_a_key(tmp_path, monkeypatch):
    config = load_config()
    config["llm"] = {**config["llm"], "provider": "ollama", "reasoning_effort": None}
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
    c = make_client(config, "generator", env_file=tmp_path / "none.env")
    assert isinstance(c.provider, OllamaProvider) and c.provider.host == "http://127.0.0.1:9999"


# --- JSON mode and is_cached --------------------------------------------------------------------


def test_json_mode_changes_the_hash_but_off_keeps_old_keys():
    base = LLMRequest("m", PROMPT, 0.0, 10)
    on = LLMRequest("m", PROMPT, 0.0, 10, json_output=True)
    assert request_hash("groq", base) != request_hash("groq", on)
    # The key of a request without JSON mode is what it was before the field existed.
    old_style = json.dumps(
        {
            "provider": "groq",
            "model": "m",
            "messages": [{"role": "user", "content": "What is RRF?"}],
            "temperature": 0.0,
            "max_output_tokens": 10,
            "reasoning_effort": None,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    import hashlib

    assert request_hash("groq", base) == hashlib.sha256(old_style.encode()).hexdigest()


def test_is_cached(tmp_path):
    c, fake, _ = client(["answer"], tmp_path)
    assert not c.is_cached(PROMPT)
    c.generate(PROMPT)
    assert c.is_cached(PROMPT) and not c.is_cached(PROMPT, temperature=0.5)
    assert len(fake.requests) == 1


def test_judge_clients_use_json_mode(tmp_path, keys):
    config = load_config()
    assert make_client(config, "judge", env_file=tmp_path / "x").defaults["json_output"] is True
    assert (
        make_client(config, "generator", env_file=tmp_path / "x").defaults["json_output"] is False
    )


# --- latency vs waiting -------------------------------------------------------------------------


def test_latency_excludes_throttle_and_backoff_waits():
    c, _, clock = client(
        ["a", RateLimited("slow", retry_after=5), "b"], limits={"rpm": 1, "tpm": None}
    )
    first = c.generate([Message("user", "one")])
    second = c.generate([Message("user", "two")])  # waits ~60 s (RPM) + ~5 s (Retry-After)
    assert first.wait_ms == 0
    assert second.wait_ms == pytest.approx(1000 * sum(clock.sleeps))
    assert second.wait_ms > 60_000
    assert second.latency_ms < 1000  # the fake API call itself is instant


def test_wait_ms_is_cached_with_the_response(tmp_path):
    c, _, _ = client(["a"], tmp_path)
    c.generate(PROMPT)
    again = c.generate(PROMPT)
    assert again.cached and again.wait_ms == 0


def test_throttle_wait_returns_the_seconds_slept():
    clock = FakeClock()
    throttle = MinuteThrottle(rpm=1, tpm=None, clock=clock, sleep=clock.sleep)
    assert throttle.wait(10) == 0  # nothing recorded yet: no sleep
    throttle.record(10)
    assert throttle.wait(10) == pytest.approx(sum(clock.sleeps)) and clock.sleeps


def test_no_wait_is_exactly_zero_on_the_real_clock():
    """No fake clock: timer overhead must not leak into wait_ms (it did on Linux)."""
    r = LLMClient(FakeProvider(["a"]), "m").generate(PROMPT)
    assert r.wait_ms == 0

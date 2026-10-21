import pytest
import requests

from ragpapers.generation.providers import (
    FakeProvider,
    GeminiProvider,
    GroqProvider,
    LLMRequest,
    Message,
    OllamaProvider,
    ProviderError,
    RateLimited,
    TransientError,
)

MESSAGES = [Message("system", "Answer briefly."), Message("user", "What is RRF?")]


def request(model="some-model", reasoning_effort=None):
    return LLMRequest(model, MESSAGES, 0.0, 256, reasoning_effort)


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code, self._body, self.headers = status_code, body, headers or {}
        self.text = str(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return self._body


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self.error:
            raise self.error
        return self.response


# --- Groq ---------------------------------------------------------------------------------------

GROQ_OK = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "openai/gpt-oss-120b",
    "system_fingerprint": "fp_abc",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Fusion.", "reasoning": "Let me think"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 40,
        "completion_tokens": 25,
        "total_tokens": 65,
        "completion_tokens_details": {"reasoning_tokens": 20},
    },
}


def test_groq_builds_an_openai_style_request_and_parses_the_reply():
    session = FakeSession(FakeResponse(200, GROQ_OK))
    provider = GroqProvider(api_key="sk-test", session=session, timeout_s=30)
    r = provider.send(request("openai/gpt-oss-120b", reasoning_effort="low"))

    (call,) = session.calls
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"] == {"Authorization": "Bearer sk-test"}
    assert call["timeout"] == 30
    assert call["json"] == {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is RRF?"},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 256,
        "reasoning_effort": "low",
    }
    assert r.text == "Fusion."  # reasoning text dropped
    assert (r.model, r.system_fingerprint, r.response_id) == (
        "openai/gpt-oss-120b",
        "fp_abc",
        "chatcmpl-123",
    )
    assert (r.input_tokens, r.output_tokens, r.reasoning_tokens) == (40, 25, 20)
    assert r.finish_reason == "stop"
    assert "sk-test" not in repr(provider) and "sk-test" not in repr(r)


def test_groq_omits_reasoning_effort_when_not_set():
    provider = GroqProvider(api_key="k", session=FakeSession())
    assert "reasoning_effort" not in provider.payload(request())


def test_groq_429_carries_retry_after_and_daily_flag():
    minute = FakeResponse(
        429,
        {"error": {"message": "Rate limit reached on tokens per minute (TPM)"}},
        {"retry-after": "7"},
    )
    with pytest.raises(RateLimited) as err:
        GroqProvider(api_key="k", session=FakeSession(minute)).send(request())
    assert err.value.retry_after == 7 and not err.value.daily

    day = FakeResponse(429, {"error": {"message": "Rate limit reached on tokens per day (TPD)"}})
    with pytest.raises(RateLimited) as err:
        GroqProvider(api_key="k", session=FakeSession(day)).send(request())
    assert err.value.daily


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (FakeResponse(401, {"error": {"message": "Invalid API Key"}}), ProviderError),
        (FakeResponse(404, {"error": {"message": "model not found"}}), ProviderError),
        (FakeResponse(503, "Service Unavailable"), TransientError),
    ],
)
def test_groq_http_errors_are_classified(response, error):
    with pytest.raises(error):
        GroqProvider(api_key="k", session=FakeSession(response)).send(request())


@pytest.mark.parametrize("exc", [requests.Timeout("slow"), requests.ConnectionError("reset")])
def test_timeouts_and_dropped_connections_are_transient(exc):
    with pytest.raises(TransientError):
        GroqProvider(api_key="k", session=FakeSession(error=exc)).send(request())


# --- Gemini -------------------------------------------------------------------------------------

GEMINI_OK = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Fu"}, {"text": "sion."}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 12,
        "candidatesTokenCount": 3,
        "thoughtsTokenCount": 30,
        "totalTokenCount": 45,
    },
    "modelVersion": "gemini-3.5-flash-lite",
    "responseId": "resp-1",
}


def test_gemini_builds_a_generate_content_request_and_parses_the_reply():
    session = FakeSession(FakeResponse(200, GEMINI_OK))
    r = GeminiProvider(api_key="g-test", session=session).send(request("gemini-3.5-flash-lite"))

    (call,) = session.calls
    assert call["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash-lite:generateContent"
    )
    assert call["headers"] == {"x-goog-api-key": "g-test"}  # key in a header, not the URL
    assert call["json"] == {
        "systemInstruction": {"parts": [{"text": "Answer briefly."}]},
        "contents": [{"role": "user", "parts": [{"text": "What is RRF?"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }
    assert r.text == "Fusion."
    assert r.model == "gemini-3.5-flash-lite" and r.response_id == "resp-1"
    assert (r.input_tokens, r.output_tokens, r.reasoning_tokens) == (12, 33, 30)
    assert r.finish_reason == "STOP"


def test_gemini_maps_assistant_role_and_skips_thought_parts():
    msgs = [Message("user", "a"), Message("assistant", "b"), Message("user", "c")]
    payload = GeminiProvider(api_key="k").payload(LLMRequest("m", msgs, 0.0, 10))
    assert [c["role"] for c in payload["contents"]] == ["user", "model", "user"]
    assert "systemInstruction" not in payload

    body = {
        "candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}, {"text": "ok"}]}}]
    }
    assert GeminiProvider(api_key="k").parse(body, request()).text == "ok"


def test_gemini_blocked_prompt_gives_empty_text_and_the_reason():
    body = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 5}}
    r = GeminiProvider(api_key="k").parse(body, request("gemini-x"))
    assert r.text == "" and r.finish_reason == "SAFETY" and r.model == "gemini-x"


def test_gemini_rejects_reasoning_effort():
    with pytest.raises(ValueError):
        GeminiProvider(api_key="k").payload(request(reasoning_effort="low"))


def test_gemini_429_reads_retry_delay_and_daily_quota():
    def quota_error(quota_id):
        return FakeResponse(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [{"quotaId": quota_id}],
                        },
                        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s"},
                    ],
                }
            },
        )

    with pytest.raises(RateLimited) as err:
        GeminiProvider(
            api_key="k",
            session=FakeSession(
                quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
            ),
        ).send(request())
    assert err.value.retry_after == 36 and not err.value.daily

    with pytest.raises(RateLimited) as err:
        GeminiProvider(
            api_key="k",
            session=FakeSession(quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")),
        ).send(request())
    assert err.value.daily


# --- Ollama -------------------------------------------------------------------------------------

OLLAMA_OK = {
    "model": "qwen3:8b",
    "created_at": "2026-10-18T10:00:00Z",
    "message": {"role": "assistant", "content": "Fusion.", "thinking": "hmm"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 30,
    "eval_count": 9,
}


def test_ollama_builds_a_non_streaming_chat_request_and_parses_the_reply():
    session = FakeSession(FakeResponse(200, OLLAMA_OK))
    r = OllamaProvider(host="http://localhost:11434/", session=session).send(request("qwen3:8b"))

    (call,) = session.calls
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["headers"] == {}
    assert call["json"]["stream"] is False
    assert call["json"]["options"] == {"temperature": 0.0, "num_predict": 256}
    assert r.text == "Fusion." and r.model == "qwen3:8b"
    assert (r.input_tokens, r.output_tokens, r.finish_reason) == (30, 9, "stop")


def test_ollama_not_running_fails_fast_with_a_hint():
    session = FakeSession(error=requests.ConnectionError("refused"))
    with pytest.raises(ProviderError, match="is it running"):
        OllamaProvider(session=session).send(request())


# --- Fake ---------------------------------------------------------------------------------------


def test_fake_provider_replays_its_script_and_records_requests():
    fake = FakeProvider(["one", RateLimited("slow down", retry_after=1)], model_reported="m-v2")
    assert fake.send(request("m")).model == "m-v2"
    with pytest.raises(RateLimited):
        fake.send(request("m"))
    assert len(fake.requests) == 2
    with pytest.raises(AssertionError):
        fake.send(request("m"))


def test_real_http_is_blocked_in_tests():
    with pytest.raises(RuntimeError, match="network access blocked"):
        GroqProvider(api_key="k").session.post("https://api.groq.com/openai/v1/models", json={})

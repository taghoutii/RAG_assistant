"""LLM providers: one small interface, one adapter per service.

Every provider does the same job: take an `LLMRequest` (model, chat messages, generation
settings), translate it to one service's HTTP API, and translate the reply back into an
`LLMResponse`. The rest of the code only ever sees these two dataclasses, so switching the
generator or the judge to another provider or model is a config change.

    GroqProvider     OpenAI-compatible chat completions (generator, fallback judge)
    GeminiProvider   Google AI Studio generateContent (judge)
    OllamaProvider   local Ollama /api/chat, non-streaming (offline fallback)
    FakeProvider     scripted replies for tests: never touches the network

Providers only translate and classify errors. Caching, throttling and retries live in
generation/client.py. Only the answer text is kept: reasoning / thinking text is dropped,
and only its token count is recorded. API keys travel in request headers and are never
stored on the response.
"""

import json
import re
from dataclasses import dataclass, field

import requests


@dataclass(frozen=True)
class Message:
    role: str  # "system", "user" or "assistant"
    content: str


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: list[Message]
    temperature: float
    max_output_tokens: int
    reasoning_effort: str | None = None  # Groq reasoning models only (e.g. gpt-oss)


@dataclass
class LLMResponse:
    text: str
    provider: str
    requested_model: str
    model: str  # the exact model ID the API reports it ran
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None  # answer + reasoning/thinking tokens
    reasoning_tokens: int | None  # reasoning / thinking tokens, when reported
    response_id: str | None = None
    system_fingerprint: str | None = None
    # Filled in by the client:
    latency_ms: float | None = None
    cached: bool = False
    request_hash: str | None = None
    created_at: str | None = None


class ProviderError(Exception):
    """A call that must not be retried (bad request, unknown model, invalid key...)."""


class RateLimited(Exception):
    """HTTP 429. `retry_after` in seconds when the service says; `daily` if a per-day quota."""

    def __init__(self, message: str, retry_after: float | None = None, daily: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily


class TransientError(Exception):
    """5xx, timeout or dropped connection: worth retrying after a pause."""


class Provider:
    name = "base"

    def send(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


def _error_text(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        return str(error.get("message", error))[:300]
    return str(error)[:300]


def _post(session, provider: str, url: str, headers: dict, payload: dict, timeout: float):
    """POST and classify the outcome; returns the parsed JSON body of a 2xx reply."""
    try:
        response = session.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout as err:
        raise TransientError(f"{provider}: timeout after {timeout}s") from err
    except requests.ConnectionError as err:
        raise TransientError(f"{provider}: connection error: {err}") from err

    if response.status_code == 429:
        return response  # the caller reads its provider-specific quota details
    if response.status_code >= 500:
        raise TransientError(f"{provider}: HTTP {response.status_code}: {_error_text(response)}")
    if response.status_code >= 400:
        raise ProviderError(f"{provider}: HTTP {response.status_code}: {_error_text(response)}")
    return response


def _seconds(value) -> float | None:
    """'7.66', '7.66s' or '36s' -> seconds; None if absent or unreadable."""
    if value is None:
        return None
    match = re.fullmatch(r"\s*([\d.]+)\s*s?\s*", str(value))
    return float(match.group(1)) if match else None


# --- Groq (OpenAI-compatible) -------------------------------------------------------------------


@dataclass
class GroqProvider(Provider):
    api_key: str = field(repr=False)
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_s: float = 120
    session: object = field(default_factory=requests.Session, repr=False)
    name = "groq"

    def payload(self, request: LLMRequest) -> dict:
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_completion_tokens": request.max_output_tokens,
        }
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        return payload

    def send(self, request: LLMRequest) -> LLMResponse:
        response = _post(
            self.session,
            self.name,
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            self.payload(request),
            self.timeout_s,
        )
        if response.status_code == 429:
            message = _error_text(response)
            # Groq names the exhausted limit in the message, e.g. "tokens per day (TPD)".
            daily = "per day" in message.lower()
            raise RateLimited(
                f"groq: {message}", _seconds(response.headers.get("retry-after")), daily
            )
        return self.parse(response.json(), request)

    def parse(self, body: dict, request: LLMRequest) -> LLMResponse:
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return LLMResponse(
            text=choice["message"].get("content") or "",  # the "reasoning" field is dropped
            provider=self.name,
            requested_model=request.model,
            model=body["model"],
            finish_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            response_id=body.get("id"),
            system_fingerprint=body.get("system_fingerprint"),
        )


# --- Gemini (Google AI Studio) ------------------------------------------------------------------


@dataclass
class GeminiProvider(Provider):
    api_key: str = field(repr=False)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 120
    session: object = field(default_factory=requests.Session, repr=False)
    name = "gemini"

    def payload(self, request: LLMRequest) -> dict:
        if request.reasoning_effort is not None:
            raise ValueError("gemini: reasoning_effort is not supported (default thinking)")
        system = [m.content for m in request.messages if m.role == "system"]
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in request.messages
            if m.role != "system"
        ]
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return payload

    def send(self, request: LLMRequest) -> LLMResponse:
        response = _post(
            self.session,
            self.name,
            f"{self.base_url}/models/{request.model}:generateContent",
            {"x-goog-api-key": self.api_key},
            self.payload(request),
            self.timeout_s,
        )
        if response.status_code == 429:
            retry_after, daily = self.quota_details(response)
            raise RateLimited(f"gemini: {_error_text(response)}", retry_after, daily)
        return self.parse(response.json(), request)

    @staticmethod
    def quota_details(response) -> tuple[float | None, bool]:
        """Retry delay and whether a per-day quota was hit, from the 429 error details."""
        try:
            details = response.json()["error"].get("details", [])
        except (ValueError, KeyError, AttributeError):
            details = []
        retry_after, daily = _seconds(response.headers.get("retry-after")), False
        for detail in details:
            if "retryDelay" in detail:
                retry_after = _seconds(detail["retryDelay"])
            for violation in detail.get("violations", []):
                daily = daily or "PerDay" in violation.get("quotaId", "")
        return retry_after, daily

    def parse(self, body: dict, request: LLMRequest) -> LLMResponse:
        candidates = body.get("candidates") or [{}]  # none at all if the prompt was blocked
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts", [])
        usage = body.get("usageMetadata") or {}
        return LLMResponse(
            # Thought parts only appear when requested; skip them anyway.
            text="".join(p.get("text", "") for p in parts if not p.get("thought")),
            provider=self.name,
            requested_model=request.model,
            model=body.get("modelVersion", request.model),
            finish_reason=candidate.get("finishReason")
            or (body.get("promptFeedback") or {}).get("blockReason"),
            input_tokens=usage.get("promptTokenCount"),
            # Gemini counts thinking apart from the answer; add it so output_tokens means the
            # same for every provider (answer + reasoning).
            output_tokens=usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            reasoning_tokens=usage.get("thoughtsTokenCount"),
            response_id=body.get("responseId"),
        )


# --- Ollama (local) -----------------------------------------------------------------------------


@dataclass
class OllamaProvider(Provider):
    host: str = "http://localhost:11434"
    timeout_s: float = 120
    session: object = field(default_factory=requests.Session, repr=False)
    name = "ollama"

    def payload(self, request: LLMRequest) -> dict:
        if request.reasoning_effort is not None:
            raise ValueError("ollama: reasoning_effort is not supported by this adapter")
        return {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }

    def send(self, request: LLMRequest) -> LLMResponse:
        try:
            response = _post(
                self.session,
                self.name,
                f"{self.host.rstrip('/')}/api/chat",
                {},
                self.payload(request),
                self.timeout_s,
            )
        except TransientError as err:
            if isinstance(err.__cause__, requests.ConnectionError):
                # Nothing listening: retrying will not start Ollama.
                raise ProviderError(f"ollama: cannot reach {self.host}; is it running?") from err
            raise
        if response.status_code == 429:  # Ollama has no quotas; treat as transient
            raise TransientError(f"ollama: HTTP 429: {_error_text(response)}")
        return self.parse(response.json(), request)

    def parse(self, body: dict, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=body["message"].get("content") or "",  # the "thinking" field is dropped
            provider=self.name,
            requested_model=request.model,
            model=body["model"],
            finish_reason=body.get("done_reason"),
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            reasoning_tokens=None,  # Ollama does not report thinking tokens separately
        )


# --- Fake (tests) -------------------------------------------------------------------------------


class FakeProvider(Provider):
    """Replays a script of replies; each item is a reply text or an exception to raise.

    Every request it receives is kept in `requests`, so tests can check what was sent.
    """

    name = "fake"

    def __init__(self, script: list, model_reported: str | None = None):
        self.script = list(script)
        self.model_reported = model_reported
        self.requests: list[LLMRequest] = []

    def send(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("FakeProvider: no scripted reply left")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        prompt = json.dumps([m.content for m in request.messages])
        return LLMResponse(
            text=item,
            provider=self.name,
            requested_model=request.model,
            model=self.model_reported or request.model,
            finish_reason="stop",
            input_tokens=len(prompt) // 4,
            output_tokens=len(item) // 4,
            reasoning_tokens=0,
        )

"""LLM client: disk cache, per-minute throttle, retries with backoff, daily-quota stop.

    client = make_client(config, "generator")          # or "judge", "judge_fallback"
    response = client.generate([Message("user", "...")])

One `generate` call:
1. Hashes the full request (provider, model, messages, every generation setting). If
   cache/llm/<provider>/<model>/<hash>.json exists, that response is returned and no API call
   is made, so re-running an experiment is free and gives identical answers. Any change to
   the prompt or a setting gives a new hash, so a stale answer is never reused.
2. Waits, if needed, to stay under the configured requests/tokens per minute.
3. Calls the provider. HTTP 429, 5xx and timeouts are retried with exponential backoff and
   jitter (honouring Retry-After); other errors fail at once. When a per-day quota is hit
   (or the service asks to wait longer than `backoff_max_s`), it raises `DailyQuotaReached`
   instead of sleeping for hours: finished calls are cached, so re-running resumes.
4. Records latency, the request hash and a timestamp, and writes the cache file atomically.
"""

import hashlib
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from ragpapers.config import REPO_ROOT
from ragpapers.generation.env import load_env
from ragpapers.generation.providers import (
    GeminiProvider,
    GroqProvider,
    LLMRequest,
    LLMResponse,
    Message,
    OllamaProvider,
    Provider,
    RateLimited,
    TransientError,
)


class DailyQuotaReached(Exception):
    """The provider's per-day quota is used up; re-run later to resume from the cache."""


def request_hash(provider: str, request: LLMRequest) -> str:
    canonical = json.dumps(
        {"provider": provider, **asdict(request)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimate_tokens(messages: list[Message]) -> int:
    """Rough upper estimate (~3 characters per token) for throttling before the call."""
    return sum(len(m.content) // 3 + 4 for m in messages)


def safe_name(model: str) -> str:
    """'openai/gpt-oss-120b' -> 'openai_gpt-oss-120b' (also safe for Ollama's 'name:tag')."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


class MinuteThrottle:
    """Keeps requests and tokens within any sliding 60-second window under the limits."""

    def __init__(self, rpm: int | None, tpm: int | None, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm = rpm, tpm
        self.clock, self.sleep = clock, sleep
        self.calls: deque[tuple[float, int]] = deque()  # (time, tokens) of recent calls

    def _forget_old(self) -> None:
        while self.calls and self.clock() - self.calls[0][0] >= 60:
            self.calls.popleft()

    def wait(self, tokens: int) -> None:
        """Sleep until one more call of about `tokens` tokens fits under both limits."""
        while True:
            self._forget_old()
            if not self.calls:
                return  # always allow one call, even if its estimate exceeds the TPM limit
            too_many = self.rpm is not None and len(self.calls) + 1 > self.rpm
            used = sum(t for _, t in self.calls)
            too_big = self.tpm is not None and used + tokens > self.tpm
            if not (too_many or too_big):
                return
            self.sleep(60 - (self.clock() - self.calls[0][0]) + 0.1)

    def record(self, tokens: int) -> None:
        self.calls.append((self.clock(), tokens))


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        reasoning_effort: str | None = None,
        limits: dict | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 6,
        backoff_initial_s: float = 2,
        backoff_max_s: float = 120,
        clock=time.monotonic,
        sleep=time.sleep,
        rng: random.Random | None = None,
    ):
        self.provider, self.model = provider, model
        self.defaults = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
        }
        limits = limits or {}
        self.throttle = MinuteThrottle(limits.get("rpm"), limits.get("tpm"), clock, sleep)
        self.cache_dir = (
            Path(cache_dir) / provider.name / safe_name(model) if cache_dir is not None else None
        )
        self.max_retries = max_retries
        self.backoff_initial_s, self.backoff_max_s = backoff_initial_s, backoff_max_s
        self.sleep = sleep
        self.rng = rng or random.Random()

    def generate(self, messages: list[Message], **params) -> LLMResponse:
        """One chat completion. `params` override the client's defaults for this call."""
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"unknown generation parameters: {sorted(unknown)}")
        request = LLMRequest(self.model, list(messages), **{**self.defaults, **params})
        key = request_hash(self.provider.name, request)

        cached = self._read_cache(key)
        if cached is not None:
            return cached

        start = time.perf_counter()
        response = self._send_with_retries(request)
        response = replace(
            response,
            latency_ms=1000 * (time.perf_counter() - start),
            cached=False,
            request_hash=key,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._write_cache(key, request, response)
        return response

    def _backoff(self, attempt: int) -> float:
        delay = min(self.backoff_max_s, self.backoff_initial_s * 2**attempt)
        return delay * self.rng.uniform(0.5, 1.0)

    def _send_with_retries(self, request: LLMRequest) -> LLMResponse:
        estimate = estimate_tokens(request.messages) + request.max_output_tokens
        for attempt in range(self.max_retries + 1):
            self.throttle.wait(estimate)
            try:
                response = self.provider.send(request)
            except RateLimited as err:
                self.throttle.record(0)  # a refused call still counts as a request
                wait = err.retry_after
                if err.daily or (wait is not None and wait > self.backoff_max_s):
                    raise DailyQuotaReached(
                        f"{err} -- daily quota reached; re-run later to resume from the cache"
                    ) from err
                if attempt == self.max_retries:
                    raise
                if wait is None:
                    wait = self._backoff(attempt)
                self.sleep(wait + self.rng.uniform(0, 1))
            except TransientError:
                self.throttle.record(0)
                if attempt == self.max_retries:
                    raise
                self.sleep(self._backoff(attempt))
            else:
                used = (response.input_tokens or 0) + (response.output_tokens or 0)
                self.throttle.record(used or estimate)
                return response
        raise AssertionError("unreachable")

    def _path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.json" if self.cache_dir is not None else None

    def _read_cache(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return replace(LLMResponse(**data["response"]), cached=True)

    def _write_cache(self, key: str, request: LLMRequest, response: LLMResponse) -> None:
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "request": {"provider": self.provider.name, **asdict(request)},
            "response": asdict(response),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache file


def build_provider(name: str, providers_cfg: dict, timeout_s: float) -> Provider:
    """The provider adapter named in the config; API keys come from the environment."""
    cfg = providers_cfg[name]
    if name == "ollama":
        host = os.environ.get(cfg["host_env"]) or cfg.get("default_host", "http://localhost:11434")
        return OllamaProvider(host=host, timeout_s=timeout_s)
    classes = {"groq": GroqProvider, "gemini": GeminiProvider}
    if name not in classes:
        raise ValueError(f"unknown provider {name!r}")
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} is not set: add it to .env (see .env.example)")
    return classes[name](api_key=api_key, base_url=cfg["base_url"], timeout_s=timeout_s)


def make_client(config: dict, role: str, env_file: str | Path = REPO_ROOT / ".env") -> LLMClient:
    """The client for `role`: "generator" (config llm), "judge" or "judge_fallback"."""
    load_env(env_file)
    if role == "generator":
        section = config["llm"]
        provider, model, limits = section["provider"], section["model"], section["limits"]
    elif role == "judge":
        section = config["judge"]
        provider, model, limits = section["provider"], section["model"], section["limits"]
    elif role == "judge_fallback":
        section = config["judge"]
        provider, model = section["fallback_provider"], section["fallback_model"]
        limits = section["fallback_limits"]
    else:
        raise ValueError(f"unknown role {role!r}")

    calls = config["llm_calls"]
    return LLMClient(
        build_provider(provider, config["providers"], calls["timeout_s"]),
        model,
        temperature=section["temperature"],
        max_output_tokens=section["max_output_tokens"],
        reasoning_effort=section.get("reasoning_effort") if role == "generator" else None,
        limits=limits,
        cache_dir=REPO_ROOT / calls["cache_dir"],
        max_retries=calls["max_retries"],
        backoff_initial_s=calls["backoff_initial_s"],
        backoff_max_s=calls["backoff_max_s"],
    )

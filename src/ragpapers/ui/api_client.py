"""What the Streamlit page needs from the API, kept out of the page so it can be tested.

The UI is a thin HTTP client: it never imports the retrieval pipeline or opens the indexes
(the local Qdrant index can be opened by one process only, and that process is the API).
"""

import os

import requests

DEFAULT_API_URL = "http://localhost:8000"
TIMEOUT_S = 180  # generation retries with backoff can take a while on the free tier
START_API = "uvicorn ragpapers.api.app:app --port 8000"


class ApiError(Exception):
    """The API could not answer; `title` and `hint` are the friendly message shown on the page.

    `kind` says what went wrong: "offline", "timeout", "quota", "no_key", "provider",
    "invalid" or "other".
    """

    def __init__(self, kind: str, title: str, hint: str):
        super().__init__(f"{title}. {hint}")
        self.kind, self.title, self.hint = kind, title, hint


def api_url() -> str:
    """The API's base URL: RAG_API_URL (set by docker-compose) or localhost:8000."""
    return os.environ.get("RAG_API_URL", DEFAULT_API_URL).rstrip("/")


def _detail(r) -> str:
    try:
        detail = r.json().get("detail", r.text)
    except (ValueError, AttributeError):
        detail = r.text
    return detail if isinstance(detail, str) else str(detail)


def error_for(status: int, detail: str) -> ApiError:
    """The friendly message for an HTTP error status of the API."""
    if status == 503 and "quota" in detail.lower():
        return ApiError(
            "quota",
            "Daily limit reached",
            "The language model's free-tier quota is used up for today. The example "
            "questions still work (their answers are cached); new questions work again "
            "tomorrow.",
        )
    if status == 503:
        return ApiError(
            "no_key",
            "The language model is not configured",
            f"{detail}. Restart the API once the key is set.",
        )
    if status == 502:
        return ApiError(
            "provider",
            "The language model provider had a problem",
            "This is usually temporary. Please try again in a minute.",
        )
    if status == 422:
        return ApiError("invalid", "Please check your question", "Use 1 to 1000 characters.")
    return ApiError("other", f"Unexpected API error ({status})", detail)


def ask(question: str, base_url: str, timeout: float = TIMEOUT_S) -> dict:
    """POST the question to /ask and return the JSON response (raises ApiError)."""
    try:
        r = requests.post(f"{base_url}/ask", json={"question": question}, timeout=timeout)
    except requests.Timeout as err:
        raise ApiError(
            "timeout",
            "The answer is taking too long",
            f"The API did not answer within {timeout:.0f} s. Please try again.",
        ) from err
    except requests.ConnectionError as err:
        raise ApiError(
            "offline",
            "The API is not running",
            f"Start it in another terminal with: {START_API}",
        ) from err
    if r.status_code == 200:
        return r.json()
    raise error_for(r.status_code, _detail(r))


def cited_sources(response: dict) -> list[dict]:
    """Each citation of the answer, joined with the text of its chunk, in first-cited order."""
    texts = {c["chunk_id"]: c["text"] for c in response["retrieved"]}
    return [{**c, "text": texts.get(c["chunk_id"], "")} for c in response["citations"]]

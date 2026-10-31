import re
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from ragpapers.ui import api_client, display

APP = Path(api_client.__file__).with_name("app.py")
ABSTAIN = "I couldn't find sufficient evidence in the available sources."


def retrieved(n):
    return {
        "rank": n,
        "chunk_id": f"p:{n:04d}",
        "paper_id": "p",
        "title": "Paper",
        "year": 2020,
        "section": f"Section {n}",
        "url": f"https://arxiv.org/html/p#S{n}",
        "text": f"chunk text {n}",
        "rrf_score": 0.01,
        "bm25_rank": n,
        "dense_rank": None,
    }


def response(answer="LoRA freezes W0 [2].", cited=(2,), abstained=False, cached=False):
    chunks = [retrieved(n) for n in range(1, 9)]
    return {
        "request_id": "r1",
        "question": "Q?",
        "answer": answer,
        "abstained": abstained,
        "citations": [
            {k: chunks[n - 1][k] for k in ("chunk_id", "paper_id", "title", "year", "section")}
            | {"number": n, "url": chunks[n - 1]["url"]}
            for n in cited
        ],
        "invalid_citations": [],
        "retrieved": chunks,
        "latency_ms": {
            "bm25": 10.0,
            "dense": 20.0,
            "fusion": 0.4,
            "generation": 700.0,
            "total": 731.0,
        },
        "model": "openai/gpt-oss-120b",
        "cached": cached,
    }


class FakeHTTPResponse:
    def __init__(self, status_code, body):
        self.status_code, self.body, self.text = status_code, body, str(body)

    def json(self):
        return self.body


# --- api_client --------------------------------------------------------------------------------


def test_ask_posts_the_question(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json)
        return FakeHTTPResponse(200, response())

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    assert api_client.ask("Q?", "http://api:8000")["answer"].startswith("LoRA")
    assert sent == {"url": "http://api:8000/ask", "json": {"question": "Q?"}}


@pytest.mark.parametrize(
    "status, detail, kind, title",
    [
        (503, "The generator's daily free-tier quota is used up", "quota", "Daily limit reached"),
        (503, "GROQ_API_KEY is not set", "no_key", "The language model is not configured"),
        (502, "The generator API failed: HTTP 500", "provider", "provider had a problem"),
        (422, [{"msg": "too short"}], "invalid", "Please check your question"),
        (500, "boom", "other", "Unexpected API error (500)"),
    ],
)
def test_api_errors_become_friendly_messages(monkeypatch, status, detail, kind, title):
    monkeypatch.setattr(
        api_client.requests, "post", lambda *a, **k: FakeHTTPResponse(status, {"detail": detail})
    )
    with pytest.raises(api_client.ApiError, match=re.escape(title)) as caught:
        api_client.ask("Q?", "http://x")
    assert caught.value.kind == kind


def test_the_quota_message_says_examples_still_work():
    err = api_client.error_for(503, "daily free-tier quota is used up")
    assert "example questions still work" in err.hint


def test_an_unreachable_api_says_how_to_start_it(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(api_client.requests, "post", refuse)
    with pytest.raises(api_client.ApiError) as caught:
        api_client.ask("Q?", "http://localhost:8000")
    assert caught.value.kind == "offline"
    assert caught.value.title == "The API is not running"
    assert "uvicorn ragpapers.api.app:app" in caught.value.hint


def test_a_timeout_is_reported(monkeypatch):
    def slow(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(api_client.requests, "post", slow)
    with pytest.raises(api_client.ApiError) as caught:
        api_client.ask("Q?", "http://x", timeout=5)
    assert caught.value.kind == "timeout" and "5 s" in caught.value.hint


def test_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("RAG_API_URL", raising=False)
    assert api_client.api_url() == "http://localhost:8000"
    monkeypatch.setenv("RAG_API_URL", "http://api:8000/")
    assert api_client.api_url() == "http://api:8000"


def test_cited_sources_carry_their_chunk_text_in_cited_order():
    sources = api_client.cited_sources(response(cited=(3, 1)))
    assert [s["number"] for s in sources] == [3, 1]
    assert sources[0]["text"] == "chunk text 3"


# --- the page ----------------------------------------------------------------------------------


@pytest.fixture
def asked(monkeypatch):
    """Replace the API call; returns the list of questions the page sent."""
    sent = []

    def install(reply):
        def fake_ask(question, base_url):
            sent.append(question)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(api_client, "ask", fake_ask)
        return sent

    return install


def page_after_typing(question):
    page = AppTest.from_file(str(APP), default_timeout=30).run()
    page.text_input(key="question").input(question)
    page.button(key="ask").click()
    return page.run()


def markdown_text(page):
    return "\n".join(m.value for m in page.markdown)


def test_the_page_has_the_hero_question_card_examples_and_sidebar():
    page = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not page.exception
    text = markdown_text(page)
    assert '<span class="hero-pill">Grounded in 44 AI/ML papers</span>' in text
    assert '<div class="hero-title">AI/ML Research Assistant</div>' in text
    assert "When the papers don't cover a question" in text
    assert '<div class="q-label">Your question</div>' in text
    assert '<div class="try-label">Try an example</div>' in text
    labels = [b.label for b in page.button if b.key and b.key.startswith("chip_")]
    assert labels == [label for label, _ in display.EXAMPLES]
    (sidebar,) = [m.value for m in page.sidebar.markdown]  # one block, nothing else
    assert '<div class="corpus-count">44 papers</div>' in sidebar
    assert sidebar.count('class="topic-row"') == 9


def test_a_chip_asks_its_dev_question_word_for_word(asked):
    sent = asked(response())
    page = AppTest.from_file(str(APP), default_timeout=30).run()
    page.button(key="chip_3").click()
    page.run()
    label, question = display.EXAMPLES[3]
    assert sent == [question]
    assert page.text_input(key="question").value == question


def test_the_answer_card_has_coloured_badges_and_one_source_card_per_citation(asked):
    asked(response(answer="A 【5†L1-L3】 and [2].", cited=(5, 2)))
    page = page_after_typing("Which weights does LoRA freeze?")
    assert not page.exception
    text = markdown_text(page)
    assert "2 sources cited" in text
    # Badge colours follow citation order: [5] first (indigo), then [2] (teal).
    assert (
        'A <span class="cite-badge" style="background:#4338ca">5</span> and '
        '<span class="cite-badge" style="background:#0f766e">2</span>.'
    ) in text
    assert "【" not in text
    # "Generated in" is latency_ms.generation (700 ms), not the request total (731 ms).
    assert "Generated in 0.70 s · openai/gpt-oss-120b · 8 passages retrieved" in text
    assert "cached-pill" not in text  # this answer was not cached
    # One source card per cited source, in citation order, same colour as its badge.
    assert '<span class="num-badge" style="background:#4338ca">5</span>' in text
    assert '<span class="num-badge" style="background:#0f766e">2</span>' in text
    assert text.index('num-badge" style="background:#4338ca">5') < text.index(
        'num-badge" style="background:#0f766e">2'
    )
    assert 'href="https://arxiv.org/html/p#S5" target="_blank"' in text
    assert '<div class="sources-heading">Sources</div>' in text
    # First passage open ("Hide passage"), the others collapsed ("Show passage").
    assert [(e.label, e.proto.expanded) for e in page.expander] == [
        ("Hide passage", True),
        ("Show passage", False),
    ]
    passages = [m.value for m in page.markdown if m.value.startswith("chunk text")]
    assert passages == ["chunk text 5", "chunk text 2"]  # Markdown (KaTeX), not raw text
    # The other 6 retrieved chunks are not displayed.
    assert "chunk text 1" not in text


def test_math_in_passages_and_answers_reaches_katex(asked):
    reply = response(answer=r"It solves \(\mathbf{X}\mathbf{W}=\mathbf{Y}\) [2].", cited=(2,))
    reply["retrieved"][1]["text"] = r"We solve ${\mathbf{X}\mathbf{W}=\mathbf{Y}}$ for W_1."
    asked(reply)
    page = page_after_typing("Q?")
    text = markdown_text(page)
    assert not page.exception
    assert r"It solves $\mathbf{X}\mathbf{W}=\mathbf{Y}$ " in text
    assert r"We solve ${\mathbf{X}\mathbf{W}=\mathbf{Y}}$ for W\_1\." in text


def test_a_cached_answer_shows_the_cached_pill(asked):
    asked(response(cached=True))
    text = markdown_text(page_after_typing("Q?"))
    assert '<span class="cached-pill">cached</span>' in text


def test_an_abstention_shows_the_not_found_card_without_sources(asked):
    asked(response(answer=ABSTAIN, cited=(), abstained=True))
    page = page_after_typing("How much did GPT-3 cost?")
    text = markdown_text(page)
    assert "Not found in the papers" in text and ABSTAIN.replace("'", "&#x27;") in text
    assert "The 8 most relevant passages were checked" in text
    assert "Generated in 0.70 s" in text  # the same footer line
    assert "sources-heading" not in text and "Answer</span>" not in text
    assert len(page.expander) == 0 and len(page.error) == 0
    assert "cite-badge" not in text


def test_an_offline_api_is_an_error_with_how_to_start_it(asked):
    asked(api_client.ApiError("offline", "The API is not running", "Start it with uvicorn"))
    page = page_after_typing("Q?")
    assert [e.value for e in page.error] == ["**The API is not running**  \nStart it with uvicorn"]
    assert len(page.expander) == 0


def test_the_daily_quota_is_a_warning_not_an_error(asked):
    asked(api_client.error_for(503, "daily free-tier quota is used up"))
    page = page_after_typing("Q?")
    assert len(page.error) == 0
    assert "Daily limit reached" in page.warning[0].value


def test_an_empty_question_is_not_sent(asked):
    sent = asked(response())
    page = page_after_typing("   ")
    assert sent == []
    assert "Please type a question first." in page.warning[0].value

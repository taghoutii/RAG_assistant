"""PaperRAG, the Streamlit page: ask a question, read the cited answer, open its sources.

Run (PowerShell, from the repo root, with the API already running):
    streamlit run src/ragpapers/ui/app.py
The page calls the API over HTTP (RAG_API_URL, default http://localhost:8000). It shows the
answer and the sources it cites; the full list of retrieved chunks stays in the /ask
response and the query log. Theme: .streamlit/config.toml + style.css.
"""

from pathlib import Path

import streamlit as st

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.ui import api_client, display, latex

STYLE = Path(__file__).with_name("style.css")
ERROR_ICONS = {
    "offline": ":material/power_off:",
    "timeout": ":material/hourglass_empty:",
    "quota": ":material/event_busy:",
    "no_key": ":material/key_off:",
    "provider": ":material/cloud_off:",
    "invalid": ":material/edit_note:",
    "other": ":material/error:",
}


@st.cache_data
def corpus() -> display.Corpus | None:
    """The corpus facts of the sidebar, read once from data/corpus_manifest.yaml."""
    return display.load_corpus(REPO_ROOT / load_config()["paths"]["manifest"])


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def choose_example(question: str) -> None:
    """Chip callback: put the example in the question box and ask it."""
    st.session_state["question"] = question
    st.session_state["pending"] = question


def show_answer(response: dict, facts: display.Corpus | None) -> None:
    if display.is_abstention_response(response):
        with st.container(key="abstain_card"):
            html(display.abstention_html(response))
            html(display.footer_html(response))
        return

    sources = api_client.cited_sources(response)
    colors = display.badge_colors([s["number"] for s in sources])
    with st.container(key="answer_card"):
        html(display.answer_header_html(len(sources)))
        with st.container(key="answer_body"):
            html(display.answer_html(response["answer"], colors))
            if response["invalid_citations"]:
                numbers = ", ".join(f"[{n}]" for n in response["invalid_citations"])
                st.caption(f"The answer also cites {numbers}, which match no retrieved passage.")
        html(display.footer_html(response))

    if sources:
        html('<div class="sources-heading">Sources</div>')
    for i, source in enumerate(sources):
        with st.container(key=f"source_{source['number']}"):
            html(display.source_head_html(source, colors[source["number"]], facts))
            first = i == 0
            with (
                st.expander("Hide passage" if first else "Show passage", expanded=first),
                st.container(key=f"passage_{source['number']}"),
            ):
                st.markdown(latex.passage_markdown(source["text"]))  # KaTeX renders the math


st.set_page_config(page_title="PaperRAG", page_icon=":material/menu_book:", layout="centered")
st.html(f"<style>{STYLE.read_text(encoding='utf-8')}</style>")
facts = corpus()

with st.sidebar:
    html(display.sidebar_html(facts))

html(display.hero_html(facts))

with st.container(key="question_card"):
    html('<div class="q-label">Your question</div>')
    with st.form("ask_form", border=False):
        field, button = st.columns([5, 1], vertical_alignment="bottom")
        question = field.text_input(
            "Your question",
            key="question",
            placeholder="e.g. How does LoRA reduce the number of trainable parameters?",
            label_visibility="collapsed",
        )
        submitted = button.form_submit_button("Ask", type="primary", key="ask", width="stretch")
    html('<div class="try-label">Try an example</div>')
    with st.container(horizontal=True, key="chips"):
        for i, (label, example) in enumerate(display.EXAMPLES):
            st.button(label, key=f"chip_{i}", on_click=choose_example, args=(example,))

pending = st.session_state.pop("pending", None)
if submitted:
    if question.strip():
        pending = question
    else:
        st.warning("Please type a question first.", icon=":material/edit_note:")

if pending:
    with st.spinner("Searching the papers..."):
        try:
            st.session_state["response"] = api_client.ask(pending, api_client.api_url())
            st.session_state.pop("error", None)
        except api_client.ApiError as err:
            st.session_state.pop("response", None)
            st.session_state["error"] = err

# Kept in the session so the answer stays on the page when the user opens a passage.
error = st.session_state.get("error")
if error is not None:
    # Temporary conditions (quota, slow or failing provider) are warnings, not errors.
    notice = st.warning if error.kind in ("quota", "timeout", "provider") else st.error
    notice(f"**{error.title}**  \n{error.hint}", icon=ERROR_ICONS.get(error.kind))

response = st.session_state.get("response")
if response:
    show_answer(response, facts)

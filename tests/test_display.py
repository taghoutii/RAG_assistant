from collections import Counter

import pytest

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.evaluation.questions import load_questions
from ragpapers.evaluation.split import load_split
from ragpapers.generation.prompt import parse_citations
from ragpapers.generation.runlog import load_done
from ragpapers.ingest.manifest import load_manifest
from ragpapers.ui import display

CONFIG = load_config()
PATHS = CONFIG["paths"]
ABSTAIN = CONFIG["abstention_text"]
MANIFEST = REPO_ROOT / PATHS["manifest"]

# --- corpus facts from the manifest ------------------------------------------------------------


def test_corpus_facts_are_computed_from_the_manifest():
    papers = load_manifest(MANIFEST)
    corpus = display.load_corpus(MANIFEST)
    assert corpus.n_papers == len(papers) == 44
    assert (corpus.first_year, corpus.last_year) == (
        min(p.year for p in papers),
        max(p.year for p in papers),
    )
    assert dict(corpus.topics) == Counter(p.topic for p in papers)
    assert corpus.topic_of["1706.03762"] == "transformers"


def test_topics_are_sorted_by_count_then_name():
    topics = display.load_corpus(MANIFEST).topics
    counts = [count for _, count in topics]
    assert counts == sorted(counts, reverse=True)
    names = [display.topic_style(tag).name for tag, _ in topics]
    assert names[:2] == ["RAG & retrieval", "Instruction tuning & RLHF"]
    assert names[2:5] == sorted(names[2:5])  # three topics tie at 5 papers


def test_a_small_manifest_is_read_too(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "papers:\n"
        "  - {arxiv_id: '1', version: 1, title: A, year: 2019, topic: peft}\n"
        "  - {arxiv_id: '2', version: 1, title: B, year: 2023, topic: peft}\n"
        "  - {arxiv_id: '3', version: 1, title: C, year: 2021, topic: brand-new}\n",
        encoding="utf-8",
    )
    corpus = display.load_corpus(manifest)
    assert (corpus.n_papers, corpus.first_year, corpus.last_year) == (3, 2019, 2023)
    assert corpus.topics == [("peft", 2), ("brand-new", 1)]
    assert display.topic_style("brand-new") == display.OTHER_TOPIC


def test_a_missing_manifest_gives_no_corpus(tmp_path):
    assert display.load_corpus(tmp_path / "missing.yaml") is None


def test_every_manifest_topic_has_a_display_style():
    for tag, _ in display.load_corpus(MANIFEST).topics:
        assert tag in display.TOPIC_STYLES, f"no colour or name for topic {tag!r}"


def test_the_sidebar_shows_brand_corpus_card_and_topics_only():
    shown = display.sidebar_html(display.load_corpus(MANIFEST))
    assert "PaperRAG" in shown and "Research paper assistant" in shown
    assert '<div class="corpus-count">44 papers</div>' in shown
    assert "AI/ML research from arXiv, 2014–2025" in shown
    assert shown.count('class="topic-row"') == 9
    first_row = 'style="background:#38bdf8"></span><span class="topic-name">RAG &amp; retrieval'
    assert first_row in shown
    assert "How it works" not in shown and "github" not in shown.lower()
    assert display.sidebar_html(None).count("topic-row") == 0  # no manifest: brand only


# --- citation normalisation and badges (display only) ------------------------------------------


@pytest.mark.parametrize(
    "raw, shown",
    [
        ("Fact [3].", "Fact [3]."),
        ("Fact 【3】.", "Fact [3]."),
        ("Fact 【3, 5】 and 【4,6】.", "Fact [3, 5] and [4, 6]."),
        ("Fact 【3†L1-L4】.", "Fact [3]."),
        ("Fact 【2†source】【7】.", "Fact [2][7]."),
        ("Mixed [3】 pair.", "Mixed [3] pair."),
        ("Not citations: [a], (3), [3a], [].", "Not citations: [a], (3), [3a], []."),
    ],
)
def test_every_parser_form_is_shown_as_plain_brackets(raw, shown):
    assert display.normalize_citations(raw) == shown


@pytest.mark.parametrize("raw", ["A 【3】 b 【1†L2】 c [2, 8]", "x 【5,1】 y 【9】"])
def test_normalising_keeps_the_numbers_the_parser_reads(raw):
    assert parse_citations(display.normalize_citations(raw), 8) == parse_citations(raw, 8)


def test_badge_colours_cycle_in_citation_order():
    colors = display.badge_colors([5, 2, 7, 1])
    assert colors == {5: "#4338ca", 2: "#0f766e", 7: "#6d28d9", 1: "#4338ca"}


def badge(n, color):
    return f'<span class="cite-badge" style="background:{color}">{n}</span>'


def test_citations_become_coloured_badges():
    colors = display.badge_colors([2, 1, 3])
    shown = display.answer_html("A 【2†L1】 and B [1, 3]. Again [2].", colors)
    assert shown == (
        f"A {badge(2, '#4338ca')} and B {badge(1, '#0f766e')}{badge(3, '#6d28d9')}. "
        f"Again {badge(2, '#4338ca')}."
    )


def test_numbers_that_are_not_valid_citations_stay_plain_text():
    assert display.answer_html("A [9] and [2, 12].", {2: "#4338ca"}) == "A [9] and [2, 12]."


def test_the_answer_is_not_changed_by_display():
    response = {"answer": "Fact 【3†L1-L4】.", "abstained": False}
    display.answer_html(response["answer"], {3: "#4338ca"})
    assert response["answer"] == "Fact 【3†L1-L4】."


def test_model_text_is_html_escaped():
    shown = display.answer_html("<script>x</script> & 【1】", {1: "#4338ca"})
    assert "<script>" not in shown
    assert shown.startswith("&lt;script&gt;x&lt;/script&gt; &amp; ")


# --- source cards -------------------------------------------------------------------------------


def source(**overrides):
    return {
        "number": 2,
        "chunk_id": "1706.03762:0004",
        "paper_id": "1706.03762",
        "title": "Attention Is All You Need",
        "year": 2017,
        "section": "3 Model Architecture > 3.2 Attention",
        "url": "https://arxiv.org/html/1706.03762v7#S3.SS2",
        **overrides,
    }


def test_a_source_head_has_badge_title_section_topic_and_link():
    shown = display.source_head_html(source(), "#0f766e", display.load_corpus(MANIFEST))
    assert '<span class="num-badge" style="background:#0f766e">2</span>' in shown
    assert 'Attention Is All You Need <span class="src-year">(2017)</span>' in shown
    assert "3 Model Architecture › 3.2 Attention" in shown
    assert 'style="background:#fef3c7;color:#92400e">Transformers &amp; attention</span>' in shown
    assert 'href="https://arxiv.org/html/1706.03762v7#S3.SS2" target="_blank"' in shown
    assert "Open on arXiv</a>" in shown


def test_a_source_head_is_escaped_and_works_without_a_corpus():
    shown = display.source_head_html(source(title="A <b> & C", paper_id="x"), "#4338ca", None)
    assert "A &lt;b&gt; &amp; C" in shown
    assert "topic-pill" not in shown


# --- abstention, header and footer --------------------------------------------------------------


def test_the_api_flag_marks_an_abstention():
    assert display.is_abstention_response({"answer": "anything", "abstained": True})


def test_the_abstention_sentence_alone_is_an_abstention():
    assert display.is_abstention_response({"answer": ABSTAIN, "abstained": False})
    curly = "  " + ABSTAIN.replace("'", "’") + "\n"
    assert display.is_abstention_response({"answer": curly, "abstained": False})


def test_an_answer_is_not_an_abstention():
    assert not display.is_abstention_response({"answer": "Fact [1].", "abstained": False})
    partial = f"Fact [1]. {ABSTAIN}"
    assert not display.is_abstention_response({"answer": partial, "abstained": False})


def test_the_abstention_card_text():
    shown = display.abstention_html({"retrieved": [{}] * 8})
    assert "Not found in the papers" in shown
    assert "I couldn&#x27;t find sufficient evidence in the available sources." in shown
    assert "The 8 most relevant passages were checked and none answers this question." in shown


def test_the_answer_header_counts_sources():
    assert "1 source cited" in display.answer_header_html(1)
    assert "3 sources cited" in display.answer_header_html(3)


def footer_response(generation_ms, cached, total_ms=12278.0):
    return {
        "latency_ms": {
            "bm25": 9.0,
            "dense": 10835.0,
            "generation": generation_ms,
            "total": total_ms,
        },
        "model": "openai/gpt-oss-120b",
        "retrieved": [{}] * 8,
        "cached": cached,
    }


def test_generated_in_is_the_generation_call_not_the_request_total():
    # A first question after the API starts: ~10.8 s loading the embedder, 1.4 s generating.
    shown = display.footer_html(footer_response(1434.0, cached=False))
    assert shown == (
        '<div class="card-footer">Generated in 1.4 s · openai/gpt-oss-120b · '
        "8 passages retrieved</div>"
    )
    assert "12 s" not in shown


def test_a_cached_answer_has_a_near_zero_generation_time_and_the_cached_pill():
    cached = display.footer_html(footer_response(0.6, cached=True, total_ms=44.0))
    assert "Generated in &lt; 0.01 s" in cached  # "<" HTML-escaped; shown as "<"
    assert cached.endswith('<span class="cached-pill">cached</span></div>')


@pytest.mark.parametrize(
    "ms, shown",
    [(0.6, "< 0.01 s"), (44, "0.04 s"), (834, "0.83 s"), (1834, "1.8 s"), (12278, "12 s")],
)
def test_seconds(ms, shown):
    assert display.seconds(ms) == shown


# --- examples ----------------------------------------------------------------------------------


def test_examples_are_answerable_dev_questions_word_for_word_and_never_test_questions():
    questions = {q["question"]: q for q in load_questions(REPO_ROOT / PATHS["eval_questions"])}
    dev = load_split(REPO_ROOT / PATHS["eval_splits"], "dev")
    test = load_split(REPO_ROOT / PATHS["eval_splits"], "test")
    assert len(display.EXAMPLES) == 6
    for label, question in display.EXAMPLES:
        assert question in questions, f"{label!r} is not an eval question word for word"
        q = questions[question]
        assert q["id"] in dev and q["id"] not in test, f"{label!r} is not a dev question"
        assert q["type"] != "unanswerable", f"{label!r} is unanswerable"
    assert len({q for _, q in display.EXAMPLES}) == len(display.EXAMPLES)


def test_no_example_abstained_in_the_recorded_hybrid_dev_run():
    results = REPO_ROOT / PATHS["results_dir"] / "generation" / "dev" / "hybrid.answers.jsonl"
    answers = load_done(results)
    ids = {q["question"]: q["id"] for q in load_questions(REPO_ROOT / PATHS["eval_questions"])}
    for label, question in display.EXAMPLES:
        record = answers[ids[question]]
        assert record["abstained"] is False, f"{label!r}: the recorded answer abstained"
        assert record["cited"], f"{label!r}: the recorded answer cites nothing"

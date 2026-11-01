import json

import pytest

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.ui import display, latex

CONFIG = load_config()


def kinds(text, brackets=False):
    return [(p.kind, p.text) for p in latex.split_math(text, brackets)]


# --- finding formulas --------------------------------------------------------------------------


def test_inline_formulas_are_found_between_prose():
    assert kinds("a $x^{2}$ b") == [("text", "a "), ("math", "x^{2}"), ("text", " b")]


def test_the_reported_formula_renders():
    pieces = latex.split_math(r"the system ${\mathbf{X}\mathbf{W}=\mathbf{Y}}$ is solved")
    assert [p.kind for p in pieces] == ["text", "math", "text"]
    assert pieces[1].text == r"{\mathbf{X}\mathbf{W}=\mathbf{Y}}"


def test_display_math():
    assert kinds("see $$a+b$$ here") == [("text", "see "), ("display", "a+b"), ("text", " here")]


def test_a_chunk_that_starts_inside_a_formula_pairs_the_dollars_correctly():
    # The first "$" closes a formula cut by the chunk boundary; pairing from it would turn
    # the prose " is the weight and " into "math".
    text = "W_{1}$ is the weight and $b$ is the bias"
    assert [p.text for p in latex.split_math(text) if p.kind == "math"] == ["b"]
    assert latex.split_math(text)[0] == latex.Piece("text", "W_{1}$ is the weight and ")


def test_an_unpaired_dollar_stays_text():
    assert kinds("costs $5 in total") == [("text", "costs $5 in total")]


def test_amounts_of_money_are_not_math():
    assert [p.kind for p in latex.split_math("between $5 and $10 per hour")] == [
        "text",
        "raw",
        "text",
    ]


def test_brackets_count_as_math_only_when_asked():
    text = r"constants \(c_{2}^{\text{FP32}}\) and \[x=1\]"
    assert kinds(text, brackets=True) == [
        ("text", "constants "),
        ("math", r"c_{2}^{\text{FP32}}"),
        ("text", " and "),
        ("display", "x=1"),
    ]
    assert kinds(text) == [("text", text)]


# --- what KaTeX gets ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tex, cleaned",
    [
        (r"\argmax_{y} p(y)", r"\operatorname*{arg\,max}_{y} p(y)"),
        (r"\argmin_{\theta} L", r"\operatorname*{arg\,min}_{\theta} L"),
        (r"\textsc{BLEU}", r"\text{BLEU}"),
        (r"\nicefrac{1}{2}", r"\tfrac{1}{2}"),
        (r"\mathbbm{1}", r"\mathbb{1}"),
        (r"x\textsuperscript{th}", r"x^\text{th}"),
        ("a\n  b", "a b"),
        ("a<b>c", r"a\lt b\gt c"),
        (r"\argmaxed", r"\argmaxed"),  # only the whole command is rewritten
    ],
)
def test_clean_tex(tex, cleaned):
    assert latex.clean_tex(tex) == cleaned


@pytest.mark.parametrize(
    "tex",
    [
        r"\frac{a}{b}",
        r"\left(x_{1},\cdots,x_{T}\right)",
        r"\begin{cases}1&x>0\\0&\text{else}\end{cases}",
        r"\{a, b\}",
        r"\text{softmax}(QK^{T})",
    ],
)
def test_renderable_formulas_pass(tex):
    assert latex.formula_ok(tex)


@pytest.mark.parametrize(
    "tex",
    [
        "",
        r"\frac{a}{b",
        r"a}{b",
        r"\left(x",
        r"\begin{cases}x\end{array}",
        r"\pagecolor{white}x",
        r"\penalty 10000 x",
        " is the weight and ",
        "5 and ",
    ],
)
def test_formulas_that_would_not_render_fail(tex):
    assert not latex.formula_ok(tex)


# --- rendering ---------------------------------------------------------------------------------


def test_a_broken_formula_is_shown_as_plain_text_in_a_passage():
    shown = latex.passage_markdown(r"see $\frac{a}{b$ there")
    assert r"` $\frac{a}{b$ `" in shown
    assert "see" in shown and "there" in shown


def test_passage_prose_is_literal_and_formulas_are_math():
    shown = latex.passage_markdown("a*b_c costs $5.\nNext $x_{i}$ line")
    assert shown.startswith(r"a\*b\_c costs \$5\.")
    assert "\n\nNext " in shown  # a line break becomes a paragraph break
    assert "$x_{i}$" in shown


def test_a_code_span_survives_backticks():
    assert latex.code_span("a `b` c") == "`` a `b` c ``"


def test_answer_math_is_rendered_next_to_citation_badges():
    shown = display.answer_html(r"Constants \(c_{2}\) are quantized 【1】.", {1: "#4338ca"})
    assert shown == (
        'Constants $c_{2}$ are quantized <span class="cite-badge" '
        'style="background:#4338ca">1</span>.'
    )


def test_an_answer_formula_that_would_not_render_is_plain_text():
    shown = display.answer_html(r"Uses $\pagecolor{x} a$ here [1].", {1: "#4338ca"})
    assert r"<code>$\pagecolor{x} a$</code>" in shown
    assert 'class="cite-badge"' in shown


def test_a_dollar_amount_in_an_answer_does_not_start_math():
    assert display.answer_html("It cost $5 million.", {}) == r"It cost \$5 million."


def test_html_in_answer_prose_is_still_escaped():
    assert display.answer_html("<b>x</b> $y$", {}) == "&lt;b&gt;x&lt;/b&gt; $y$"


# --- the whole corpus (local data only) ---------------------------------------------------------

CHUNKS = REPO_ROOT / CONFIG["paths"]["chunks_dir"] / "tokens448_ov64.jsonl"


@pytest.mark.skipif(not CHUNKS.exists(), reason="local chunks not built")
def test_every_corpus_passage_renders_without_error():
    for line in CHUNKS.open(encoding="utf-8"):
        text = json.loads(line)["text"]
        pieces = latex.split_math(text)
        assert "".join(
            p.text if p.kind in ("text", "raw") else "" for p in pieces
        ) or text.strip().startswith("$")
        for piece in pieces:
            if piece.kind in ("math", "display"):
                assert latex.formula_ok(piece.text)
        latex.passage_markdown(text)

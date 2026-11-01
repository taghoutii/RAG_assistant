"""Math in passages and answers: find the formulas, let KaTeX render them, never break the page.

Streamlit's st.markdown renders `$...$` (inline) and `$$...$$` (display) with KaTeX. This
module splits a text into prose and formulas so that:
- prose is shown as written (Markdown-escaped in passages, so "*" or "_" stay characters);
- a formula KaTeX can render is passed as math, after a few rewrites of commands the
  parsed papers use but KaTeX lacks (e.g. \\argmax, \\textsc);
- a formula that would not render (unknown command, unbalanced braces, or prose caught
  between two "$" because a chunk starts mid-formula) is shown as plain text instead.

Display only: the stored passage and answer are not changed.
"""

import re
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class Piece:
    kind: str  # "text", "math" (inline), "display", or "raw" (a formula shown as plain text)
    text: str  # text: the prose; math/display: the TeX without delimiters; raw: as written


# Commands of the parsed papers that KaTeX does not know, rewritten to ones it does.
REWRITES = [
    (re.compile(r"\\argmax(?![A-Za-z])"), r"\operatorname*{arg\,max}"),
    (re.compile(r"\\argmin(?![A-Za-z])"), r"\operatorname*{arg\,min}"),
    (re.compile(r"\\textsc(?![A-Za-z])"), r"\text"),
    (re.compile(r"\\nicefrac(?![A-Za-z])"), r"\tfrac"),
    (re.compile(r"\\mathbbm(?![A-Za-z])"), r"\mathbb"),
    (re.compile(r"\\textsuperscript(?![A-Za-z])"), r"^\text"),
    (re.compile(r"\\Softmax(?![A-Za-z])"), r"\operatorname{Softmax}"),
]
# Commands KaTeX cannot render: a formula using one is shown as plain text.
UNSUPPORTED = {"pagecolor", "penalty", "centering", "ref", "label", "cite", "footnote", "vspace"}

_COMMAND = re.compile(r"\\([A-Za-z]+)")
_DOLLAR = re.compile(r"(?<!\\)\$")
_DISPLAY = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)
_BRACKETS = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.DOTALL)  # \[...\] and \(...\)
# Four or more words in a row, once commands and their arguments are removed: prose.
_PROSE = re.compile(r"(?:[A-Za-z]{2,}[ ,.;:]+){3}[A-Za-z]{2,}")


def clean_tex(tex: str) -> str:
    """The formula as KaTeX should receive it (rewrites applied, on one line)."""
    for pattern, replacement in REWRITES:
        tex = pattern.sub(lambda _m, r=replacement: r, tex)
    tex = " ".join(tex.split())
    # "<" and ">" could be read as HTML tags around the math; KaTeX has \lt and \gt.
    return tex.replace("<", r"\lt ").replace(">", r"\gt ")


def _braces_balanced(tex: str) -> bool:
    depth = 0
    for i, ch in enumerate(tex):
        if i > 0 and tex[i - 1] == "\\":
            continue  # \{ and \} are literal braces
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _environments_matched(tex: str) -> bool:
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]*)\}", tex):
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def formula_ok(tex: str) -> bool:
    """True if KaTeX can be expected to render `tex` (after clean_tex) as a formula."""
    if not tex.strip():
        return False
    if any(cmd in UNSUPPORTED for cmd in _COMMAND.findall(tex)):
        return False
    if not _braces_balanced(tex) or not _environments_matched(tex):
        return False
    if len(re.findall(r"\\left(?![A-Za-z])", tex)) != len(re.findall(r"\\right(?![A-Za-z])", tex)):
        return False
    without_commands = re.sub(r"\\[A-Za-z]+\*?\s*(\{[^{}]*\})?", " ", tex)
    if _PROSE.search(without_commands):
        return False  # words between two "$" that are not a formula
    # "$5 and $10": an amount of money, not math
    return not (re.match(r"\s*\d", tex) and " " in tex.strip() and "\\" not in tex)


def _pair_dollars(text: str) -> list[Piece]:
    """Split prose with single-$ formulas, choosing how the "$" pair up.

    A chunk can start inside a formula, so its first "$" may be a closing one. Both
    pairings (from the first "$", or from the second) are tried and the one with more
    renderable formulas wins; an unpaired "$" stays a literal character.
    """
    positions = [m.start() for m in _DOLLAR.finditer(text)]
    best: list[Piece] = [Piece("text", text)]
    best_score = None
    for offset in (0, 1):
        usable = positions[offset:]
        if len(usable) < 2:
            continue
        pieces, score, last = [], 0, 0
        for start, end in zip(usable[0::2], usable[1::2], strict=False):
            pieces.append(Piece("text", text[last:start]))
            tex = clean_tex(text[start + 1 : end])
            if formula_ok(tex):
                pieces.append(Piece("math", tex))
                score += 1
            else:
                pieces.append(Piece("raw", text[start : end + 1]))
                score -= 1
            last = end + 1
        pieces.append(Piece("text", text[last:]))
        if best_score is None or score > best_score:
            best, best_score = pieces, score
    return [p for p in best if p.text]


def split_math(text: str, brackets: bool = False) -> list[Piece]:
    """Prose and formula pieces of `text`, in order.

    `$$...$$` is display math; `$...$` inline math. With `brackets` (model answers),
    `\\[...\\]` and `\\(...\\)` count too; passages use "$" only, as the parser writes them.
    """
    pattern = _BRACKETS if brackets else None
    pieces: list[Piece] = []
    last = 0
    matches = list(_DISPLAY.finditer(text))
    if pattern is not None:
        matches += list(pattern.finditer(text))
    matches.sort(key=lambda m: m.start())
    for m in matches:
        if m.start() < last:
            continue  # inside an earlier match
        pieces += _pair_dollars(text[last : m.start()])
        groups = [g for g in m.groups() if g is not None]
        tex = clean_tex(groups[0])
        inline = m.group(0).startswith("\\(")
        if formula_ok(tex):
            pieces.append(Piece("math" if inline else "display", tex))
        else:
            pieces.append(Piece("raw", m.group(0)))
        last = m.end()
    pieces += _pair_dollars(text[last:])
    return pieces


def escape_markdown(text: str) -> str:
    """Prose shown literally by Markdown: every ASCII punctuation character backslash-escaped."""
    return "".join("\\" + ch if ch in string.punctuation else ch for ch in text)


def code_span(text: str) -> str:
    """`text` as inline code (a formula that will not render, shown as readable plain text)."""
    text = " ".join(text.split())
    ticks = "``" if "`" in text else "`"
    return f"{ticks} {text} {ticks}"


def math_markdown(piece: Piece) -> str:
    if piece.kind == "math":
        return f"${piece.text}$"
    return f"\n\n$$\n{piece.text}\n$$\n\n"


def passage_markdown(text: str) -> str:
    """A passage as Markdown for st.markdown: prose literal, formulas rendered by KaTeX.

    Line breaks of the passage become paragraph breaks.
    """
    out = []
    for piece in split_math(text):
        if piece.kind == "text":
            out.append(escape_markdown(piece.text).replace("\n", "\n\n"))
        elif piece.kind == "raw":
            out.append(code_span(piece.text))
        else:
            out.append(math_markdown(piece))
    return "".join(out)

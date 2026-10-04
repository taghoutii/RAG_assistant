import re
import unicodedata

# A citation that is only numbers in brackets, e.g. "[13]", "[3, 7]" or "[4-6]". Useless without
# the bibliography, which we drop, so the parser replaces it with "[ref]".
# Author-year citations ("Devlin et al., 2019") are kept as they are.
NUMERIC_CITATION = re.compile(r"^\[\s*\d+(\s*[,–-]\s*\d+)*\s*\]$")


def normalize_text(text: str) -> str:
    """Unicode-normalize, collapse whitespace, and tidy stray spaces around punctuation."""
    text = unicodedata.normalize("NFKC", text)  # ligatures (ﬁ -> fi), non-breaking spaces, ...
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\s*\)|\[\s*\]", "", text)  # empty brackets (e.g. left by removed markup)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)  # "models ." -> "models."
    text = re.sub(r"([(\[])\s+", r"\1", text)  # "( see" -> "(see"
    return re.sub(r"\s{2,}", " ", text).strip()


def is_numeric_citation(text: str) -> bool:
    return bool(NUMERIC_CITATION.match(text.strip()))

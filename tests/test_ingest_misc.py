import pytest

from ragpapers.config import REPO_ROOT
from ragpapers.ingest.clean import is_numeric_citation, normalize_text
from ragpapers.ingest.manifest import load_manifest


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a  \n b", "a b"),
        ("models [] .", "models."),
        ("see ( ) here", "see here"),
        ("the ﬁrst", "the first"),  # ligature
        ("( see this )", "(see this)"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("[13]", True), ("[3, 7]", True), ("[4-6]", True), ("[Devlin et al., 2019]", False)],
)
def test_is_numeric_citation(text, expected):
    assert is_numeric_citation(text) is expected


def test_committed_manifest_is_valid():
    papers = load_manifest(REPO_ROOT / "data" / "corpus_manifest.yaml")
    assert 30 <= len(papers) <= 50
    assert len({p.arxiv_id for p in papers}) == len(papers)


def test_manifest_rejects_duplicates(tmp_path):
    entry = '  - {arxiv_id: "1706.03762", version: 7, title: T, year: 2017, topic: t}\n'
    path = tmp_path / "m.yaml"
    path.write_text("papers:\n" + entry + entry, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path)

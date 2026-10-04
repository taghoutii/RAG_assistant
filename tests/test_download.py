"""Offline tests for the downloader: a fake session stands in for the network."""

import json

import pytest
import requests

from ragpapers.ingest.download import NoArxivHtmlError, download_paper
from ragpapers.ingest.manifest import Paper

PAPER = Paper(arxiv_id="1706.03762", version=7, title="T", year=2017, topic="t")
CFG = {"timeout_s": 1, "request_delay_s": 0}


class FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.url = "https://arxiv.org/html/1706.03762v7"


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requested = []

    def get(self, url, timeout):
        self.requested.append(url)
        if self.error is not None:
            raise self.error
        return self.response


def test_saves_html_and_meta_from_pinned_version(tmp_path):
    session = FakeSession(FakeResponse(200, "<article class='ltx_document'>ok</article>"))
    meta = download_paper(PAPER, tmp_path, session, CFG)

    assert session.requested == ["https://arxiv.org/html/1706.03762v7"]
    assert (tmp_path / "1706.03762.html").exists()
    saved = json.loads((tmp_path / "1706.03762.meta.json").read_text(encoding="utf-8"))
    assert saved == meta
    assert meta["source"] == "arxiv_html"


@pytest.mark.parametrize(
    "response",
    [FakeResponse(404, "Not found"), FakeResponse(200, "<html>No HTML for this paper</html>")],
)
def test_missing_html_fails_without_fallback(tmp_path, response):
    session = FakeSession(response)
    with pytest.raises(NoArxivHtmlError, match="no arXiv HTML"):
        download_paper(PAPER, tmp_path, session, CFG)
    assert len(session.requested) == 1  # no second source was tried
    assert list(tmp_path.iterdir()) == []  # nothing saved


def test_network_error_is_reported_clearly(tmp_path):
    session = FakeSession(error=requests.ConnectionError("offline"))
    with pytest.raises(NoArxivHtmlError, match="request to .* failed"):
        download_paper(PAPER, tmp_path, session, CFG)

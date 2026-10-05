"""Download every paper in the manifest into data/raw/.

The only source is arXiv's own HTML at the pinned version: https://arxiv.org/html/<id>v<version>
There is deliberately no fallback (no ar5iv, no PDF): a paper without arXiv HTML makes the script
fail with a clear error, and must be replaced in the manifest. PDFs are not used because their
parse loses sections, equations and tables, and ar5iv only serves the latest version.

Each paper produces <id>.html plus <id>.meta.json (source, URL, time, sha256).
Papers that already have a meta file are skipped, so the script can be re-run safely.

Usage (PowerShell, from the repo root):
    python -m ragpapers.ingest.download
"""

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from ragpapers.config import REPO_ROOT, load_config
from ragpapers.ingest.manifest import Paper, load_manifest

SOURCE = "arxiv_html"


class NoArxivHtmlError(RuntimeError):
    """Raised when arXiv has no usable HTML rendering for the pinned paper version."""


def html_url(paper: Paper) -> str:
    return f"https://arxiv.org/html/{paper.versioned_id}"


def is_latexml_html(response: requests.Response) -> bool:
    """True if the response is a successfully converted LaTeXML paper (not an error page)."""
    return response.status_code == 200 and "ltx_document" in response.text


def download_paper(paper: Paper, raw_dir: Path, session: requests.Session, cfg: dict) -> dict:
    """Fetch the arXiv HTML of one paper and save it. Returns the metadata written.

    Raises NoArxivHtmlError if the page cannot be fetched or is not a converted paper.
    """
    url = html_url(paper)
    try:
        response = session.get(url, timeout=cfg["timeout_s"])
    except requests.RequestException as err:
        raise NoArxivHtmlError(
            f"{paper.versioned_id}: request to {url} failed ({type(err).__name__})"
        ) from err
    finally:
        time.sleep(cfg["request_delay_s"])

    if not is_latexml_html(response):
        raise NoArxivHtmlError(
            f"{paper.versioned_id}: no arXiv HTML at {url} (HTTP {response.status_code}). "
            "This project has no PDF fallback: replace the paper in data/corpus_manifest.yaml."
        )

    content = response.content
    (raw_dir / f"{paper.arxiv_id}.html").write_bytes(content)
    meta = {
        "arxiv_id": paper.arxiv_id,
        "version": paper.version,
        "source": SOURCE,
        "url": url,
        "final_url": response.url,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    meta_path = raw_dir / f"{paper.arxiv_id}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    config = load_config()
    cfg = config["ingest"]
    raw_dir = REPO_ROOT / config["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    papers = load_manifest(REPO_ROOT / config["paths"]["manifest"])

    session = requests.Session()
    session.headers["User-Agent"] = cfg["user_agent"]

    errors = []
    for i, paper in enumerate(papers, start=1):
        if (raw_dir / f"{paper.arxiv_id}.meta.json").exists():
            print(f"[{i}/{len(papers)}] {paper.versioned_id} already downloaded, skipping")
            continue
        print(f"[{i}/{len(papers)}] {paper.versioned_id} {paper.title}")
        try:
            meta = download_paper(paper, raw_dir, session, cfg)
            print(f"  saved ({meta['bytes'] // 1024} KB)")
        except NoArxivHtmlError as err:
            print(f"  ERROR: {err}")
            errors.append(str(err))

    print(f"Done. {len(papers) - len(errors)}/{len(papers)} papers available.")
    if errors:
        print("\nThe corpus is incomplete. Papers without arXiv HTML:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

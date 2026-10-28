"""The frozen state of a test-split run: everything it depends on is one git commit.

The test split is used once, with the configuration fixed on dev. Before each test step,
`frozen_commit` checks that no tracked file under the frozen paths (code, config, eval set,
corpus manifest) differs from the current commit, and returns that commit's hash; every test
result file records it. Result files written by the run itself (reports/) are not frozen.
"""

import subprocess

from ragpapers.config import REPO_ROOT

FROZEN_PATHS = ("src", "configs", "eval", "data/corpus_manifest.yaml")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def frozen_commit() -> str:
    """The current commit hash; SystemExit if any frozen file has uncommitted changes."""
    changed = _git("status", "--porcelain", "--untracked-files=no", "--", *FROZEN_PATHS)
    if changed:
        raise SystemExit(
            "test runs need the frozen state: commit or revert these changes first:\n" + changed
        )
    return _git("rev-parse", "HEAD")


def require_final_flag(split: str, final_test_run: bool) -> str | None:
    """None on dev. On test: refuse without --final-test-run, else the frozen commit hash."""
    if split != "test":
        return None
    if not final_test_run:
        raise SystemExit("the test split is held out: add --final-test-run for the final run")
    return frozen_commit()

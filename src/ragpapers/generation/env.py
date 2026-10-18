"""Read KEY=VALUE lines from a .env file into the environment (no extra dependency).

Variables already set in the environment win, so a key exported in the shell overrides .env.
Values are never printed or logged.
"""

import os
from pathlib import Path


def load_env(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

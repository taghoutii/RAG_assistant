"""JSONL query log: one JSON object per line, one line per /ask request (answered or failed).

Each line holds the question, the retrieved chunk IDs with their RRF scores and BM25 / dense
ranks, what the answer cited, the latencies and the model.
"""

import json
import threading
from pathlib import Path


class QueryLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()  # one writer at a time, so lines never interleave

    def append(self, record: dict) -> None:
        """Write `record` as one line and flush it, so a crash loses at most that line."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()


def read_log(path: str | Path) -> list[dict]:
    """Every record of a query log, oldest first (an empty list if there is no log yet)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

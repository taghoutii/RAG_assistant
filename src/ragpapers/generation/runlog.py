"""Resumable runs: one JSONL line per finished question, appended as soon as it is done.

A run that stops halfway (daily quota, crash, Ctrl+C) is restarted with the same command:
`load_done` returns the questions already in the file, and the runner skips them. A last line
cut off by a crash is ignored (that question is simply redone, from the LLM cache).
"""

import json
import os
from pathlib import Path


def load_done(path: str | Path) -> dict[str, dict]:
    """Records already written, by question ID."""
    path = Path(path)
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial last line from an interrupted write
        done[record["id"]] = record
    return done


def append_record(path: str | Path, record: dict) -> None:
    """Append one record (must have an "id") and flush it to disk immediately."""
    if "id" not in record:
        raise ValueError("record needs an 'id'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

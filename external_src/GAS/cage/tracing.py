from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class CAGETraceWriter:
    """Append-only JSONL writer for CAGE episode and optional step traces."""

    def __init__(self, path: str, debug: bool = False):
        self.path = path
        self.debug = bool(debug)
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
        self._fh.flush()

    def write_episode(self, record: dict[str, Any]) -> None:
        row = dict(record)
        row.setdefault("record_type", "episode")
        self.write(row)

    def write_step(self, record: dict[str, Any]) -> None:
        if not self.debug:
            return
        row = dict(record)
        row.setdefault("record_type", "step")
        self.write(row)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

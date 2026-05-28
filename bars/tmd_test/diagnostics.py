from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_json


def write_failure(path: str | Path, stage: str, error: BaseException, extra: dict[str, Any] | None = None) -> None:
    payload = {"status": "failed", "stage": stage, "error_type": type(error).__name__, "error": str(error)}
    if extra:
        payload.update(extra)
    write_json(path, payload)

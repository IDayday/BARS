from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Optional

from .logging import CSVLogger


@contextmanager
def phase_timer(logger: Optional[CSVLogger], phase: str, event: str, **base: Dict):
    """CSV-friendly phase timer.

    Logs a start and completed row to logs/profile.csv.  It is intentionally
    tiny and dependency-free so that profiling can stay enabled during all
    sweeps without changing runtime behavior.
    """
    t0 = time.time()
    if logger is not None:
        logger.log({'phase': phase, 'event': f'{event}_start', **base})
    try:
        yield
    finally:
        if logger is not None:
            logger.log({'phase': phase, 'event': f'{event}_end', 'duration_sec': time.time() - t0, **base})

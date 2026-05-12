"""BARS package.

Default CPU thread caps are intentionally conservative. Graph construction uses
SciPy/sklearn linear algebra; launching many experiments on a multi-GPU server
without caps can oversubscribe CPU threads and turn a minute-scale graph build
into hour-scale contention. Users can override these before importing bars.
"""
from __future__ import annotations

import os

for _k in [
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
]:
    os.environ.setdefault(_k, '1')

__version__ = '0.2.0'

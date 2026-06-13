#!/usr/bin/env python
"""Run the Phase 3 GCBC train/probe/eval pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_steps", type=int, default=0)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[phase3]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable
    if args.smoke_steps > 0:
        _run(
            [
                py,
                "scripts/train_phase3_gcbc.py",
                "--config",
                args.config,
                "--num_steps",
                str(args.smoke_steps),
            ]
        )
    if not args.skip_train:
        _run([py, "scripts/train_phase3_gcbc.py", "--config", args.config])
    _run([py, "scripts/eval_phase3_edge_execution.py", "--config", args.config, "--probe_reset_only"])
    if not args.skip_eval:
        _run([py, "scripts/eval_phase3_edge_execution.py", "--config", args.config])


if __name__ == "__main__":
    main()

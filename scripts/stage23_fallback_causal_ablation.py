#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-root", default="reports")
    args = p.parse_args()
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "stage23_fallback_causal.csv"
    if not out.exists():
        pd.DataFrame(columns=["env", "seed", "task_id", "episode_id", "trigger_state_hash", "condition", "success", "steps"]).to_csv(out, index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage27_gas.diagnostics import compare_to_baseline, load_eval_csvs, summarize_eval_rows, write_markdown_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Stage27 GAS evaluation rows")
    p.add_argument("--eval", nargs="+", required=True, help="Eval CSV paths or dirs")
    p.add_argument("--graph-summary", default=None, help="Optional graph_summary.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline", default="GAS_BASE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_eval_csvs(args.eval)
    df.to_csv(out_dir / "eval_rows_merged.csv", index=False)
    summary = summarize_eval_rows(df, group_cols=["env", "run_episodes", "variant"], success_col="success")
    summary = compare_to_baseline(summary, baseline_variant=args.baseline)
    summary.to_csv(out_dir / "stage27_summary.csv", index=False)

    graph_summary = pd.read_csv(args.graph_summary) if args.graph_summary else None
    write_markdown_report(summary, graph_summary, out_dir / "stage27_report.md")
    print(summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

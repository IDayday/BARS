#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.bridge_dataset import load_bridge_dataset
from bars.gas_bars.bridge_verifier import train_bridge_verifier


def _merge_report(existing_path: Path, df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if existing_path.exists() and len(df):
        try:
            old = pd.read_csv(existing_path)
            if set(key_cols).issubset(old.columns) and set(key_cols).issubset(df.columns):
                old_key = old[key_cols].astype(str).agg("\t".join, axis=1)
                new_key = df[key_cols].astype(str).agg("\t".join, axis=1)
                old = old.loc[~old_key.isin(set(new_key))].copy()
                df = pd.concat([old, df], ignore_index=True)
                df = df.sort_values(key_cols).reset_index(drop=True)
        except Exception:
            pass
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--reports-root", default="reports")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    rows = []
    for env in [x for x in args.envs.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            root = Path(args.artifact_root) / env / f"seed{seed}"
            edge_exec = root / "edge_exec" / "edge_exec_results.csv"
            bridge_table = root / "bridge_graphs" / "bridge_table.csv"
            if not edge_exec.exists():
                continue
            ds = load_bridge_dataset(edge_exec, bridge_table, seed=seed)
            out_dir = root / "p_bridge"
            metrics = train_bridge_verifier(ds, out_dir, epochs=args.epochs, seed=seed, device=args.device)
            metrics["env"] = env
            metrics["seed"] = seed
            rows.append(metrics)
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    df = _merge_report(reports / "stage23_p_bridge_metrics.csv", pd.DataFrame(rows), ["env", "seed"])
    df.to_csv(reports / "stage23_p_bridge_metrics.csv", index=False)


if __name__ == "__main__":
    main()

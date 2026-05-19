#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.failure_atlas import enrich_failure_atlas, grouped_failure_atlas, write_failure_report


def _read_evals(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        if root.is_file():
            frames.append(pd.read_csv(root))
            continue
        for path in root.rglob("eval.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if not {"env", "seed", "success"}.issubset(df.columns):
                continue
            df["eval_path"] = str(path)
            if "fallback_mode" not in df:
                for part in path.parts:
                    if part.startswith("fallback_"):
                        df["fallback_mode"] = part.replace("fallback_", "")
                        break
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _edge_table_for_env(artifact_root: Path, env: str, seed: int, *, include_bridges: bool = False) -> pd.DataFrame | None:
    graph_dir = artifact_root / env / f"seed{seed}" / "bridge_graphs"
    bridge = graph_dir / "bridge_table.csv"
    gas = graph_dir / "gas_graph_edges.csv"
    if include_bridges and bridge.exists() and bridge.stat().st_size > 2:
        try:
            b = pd.read_csv(bridge)
            if gas.exists():
                g = pd.read_csv(gas)
                if "edge_type" not in g:
                    g["edge_type"] = g.get("edge_source", "safe_local")
                return pd.concat([g, b], ignore_index=True).drop_duplicates("edge_id", keep="last")
            return b
        except Exception:
            return None
    if gas.exists():
        return pd.read_csv(gas)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-roots", default="runs_stage23_atlas,runs_stage23_repro,runs_stage23_integrated,runs_stage23_key_claim")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--reports-root", default="reports")
    args = p.parse_args()
    roots = [Path(x) for x in args.eval_roots.split(",") if x]
    eval_df = _read_evals(roots)
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    if len(eval_df) == 0:
        pd.DataFrame().to_csv(reports / "stage23_failure_atlas.csv", index=False)
        pd.DataFrame().to_csv(reports / "stage23_failure_atlas_grouped.csv", index=False)
        write_failure_report(pd.DataFrame(), pd.DataFrame(), reports / "stage23_failure_atlas.md")
        return
    atlases = []
    for keys, sub in eval_df.groupby([c for c in ["env", "seed", "variant"] if c in eval_df.columns], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip([c for c in ["env", "seed", "variant"] if c in eval_df.columns], keys))
        env = str(key_map.get("env"))
        seed = int(key_map.get("seed"))
        variant = str(key_map.get("variant", ""))
        include_bridges = variant not in {"gas_shortest", "official_gas_shortest_G0", "official_gas_shortest"}
        edge_table = _edge_table_for_env(Path(args.artifact_root), env, seed, include_bridges=include_bridges)
        atlases.append(enrich_failure_atlas(sub, edge_table=edge_table))
    atlas = pd.concat(atlases, ignore_index=True)
    grouped = grouped_failure_atlas(atlas)
    atlas.to_csv(reports / "stage23_failure_atlas.csv", index=False)
    grouped.to_csv(reports / "stage23_failure_atlas_grouped.csv", index=False)
    write_failure_report(atlas, grouped, reports / "stage23_failure_atlas.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.bridge_verifier import score_bridge_table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    for env in [x for x in args.envs.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            root = Path(args.artifact_root) / env / f"seed{seed}"
            model = root / "p_bridge" / "p_bridge.pt"
            bridge_table = root / "bridge_graphs" / "bridge_table.csv"
            if not model.exists() or not bridge_table.exists():
                continue
            edge_table = pd.read_csv(bridge_table)
            # Use a neutral dummy success column only to pass feature construction.
            if "success" not in edge_table:
                edge_table["success"] = 0
            score_bridge_table(model, edge_table, root / "p_bridge" / "bridge_scores.csv", device=args.device)


if __name__ == "__main__":
    main()

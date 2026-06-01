#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage27_gas.config import CalibratorConfig, NodeSelectConfig, resolve_variants
from stage27_gas.dataset import load_offline_dataset_npz
from stage27_gas.diagnostics import summarize_graph
from stage27_gas.exec_calibrator import ExecutionCalibrator, build_pair_training_set
from stage27_gas.graph_builder import build_stage27_graph
from stage27_gas.node_selection import select_stage27_nodes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Stage27 Adaptive/Execution-Calibrated GAS graphs")
    p.add_argument("--dataset", required=True, help="Input offline dataset .npz")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--variants", nargs="*", default=None, help="Subset of variants; default all")
    p.add_argument("--max-nodes", type=int, default=2500)
    p.add_argument("--coverage-k", type=int, default=1000)
    p.add_argument("--te-quantile", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-calibrator", action="store_true", help="Train p_exec calibrator for C/B4 variants")
    p.add_argument("--calibrator-path", default=None, help="Load/save calibrator pickle")
    p.add_argument("--calib-horizon", type=int, default=20)
    p.add_argument("--random-negatives", type=int, default=20000)
    p.add_argument("--hard-negatives", type=int, default=20000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_offline_dataset_npz(args.dataset)

    node_cfg = NodeSelectConfig(
        max_nodes=args.max_nodes,
        coverage_k=args.coverage_k,
        te_quantile=args.te_quantile,
        fps_seed=args.seed,
    )
    node_indices, pools = select_stage27_nodes(dataset, node_cfg)
    pd.DataFrame({"node_indices": node_indices}).to_csv(out_dir / "selected_nodes.csv", index=False)
    pool_report = {name: int(len(v)) for name, v in pools.items()}
    (out_dir / "node_pools.json").write_text(json.dumps(pool_report, indent=2), encoding="utf-8")

    variants = resolve_variants(args.variants)
    needs_calibrator = args.train_calibrator or any(
        cfg.lambda_exec > 0 or cfg.exec_gate_threshold is not None or cfg.use_tmd_gated_shortcut for cfg in variants.values()
    )
    calibrator = None
    calib_path = Path(args.calibrator_path) if args.calibrator_path else out_dir / "execution_calibrator.pkl"
    if needs_calibrator:
        if calib_path.exists() and not args.train_calibrator:
            calibrator = ExecutionCalibrator.load(calib_path)
        else:
            calib_cfg = CalibratorConfig(
                horizon=args.calib_horizon,
                seed=args.seed,
                random_negatives=args.random_negatives,
                hard_negatives=args.hard_negatives,
            )
            train_set = build_pair_training_set(dataset, calib_cfg)
            calibrator = ExecutionCalibrator()
            metrics = calibrator.fit(train_set, calib_cfg)
            calibrator.save(calib_path)
            (out_dir / "calibrator_metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")

    graph_rows = []
    for name, cfg in variants.items():
        use_cal = calibrator if (cfg.lambda_exec > 0 or cfg.exec_gate_threshold is not None or cfg.use_tmd_gated_shortcut) else None
        graph = build_stage27_graph(dataset, node_indices, cfg, calibrator=use_cal)
        graph_path = out_dir / f"graph_{name}.npz"
        graph.to_npz(graph_path)
        row = summarize_graph(graph)
        row["graph_path"] = str(graph_path)
        graph_rows.append(row)
        print(f"built {name}: nodes={graph.num_nodes} edges={graph.num_edges} path={graph_path}")

    pd.DataFrame(graph_rows).to_csv(out_dir / "graph_summary.csv", index=False)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()

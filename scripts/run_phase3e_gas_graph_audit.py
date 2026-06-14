#!/usr/bin/env python
"""Run Phase 3E GAS/TDR-style graph risk audit without environment rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.clustering import fit_state_clusters  # noqa: E402
from phase1.data import load_ogbench_dataset  # noqa: E402
from phase2.compatibility import compute_edge_compatibility  # noqa: E402
from phase3.evaluation import load_phase2_artifacts  # noqa: E402
from phase3e.gas_graph_audit import (  # noqa: E402
    audit_summary,
    bottleneck_audit,
    build_audit_edge_sets,
    edge_provenance_audit,
    path_risk_audit,
)


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "output_dir": "results/phase3e_gas_audit",
        "max_transitions": 200000,
        "cluster_method": "kmeans",
        "state_dims": None,
        "n_clusters": 512,
        "knn_k": 10,
        "seed": 0,
        "compat_H_intra": 10,
        "max_queries": 500,
        "bottleneck_top_q": 0.1,
        "edge_certification_csv": None,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    for required in ["dataset_name", "phase2_run_dir"]:
        if not merged.get(required):
            raise ValueError(f"--{required} is required")
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--cluster_method", default=None)
    parser.add_argument("--state_dims", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--knn_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compat_H_intra", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--bottleneck_top_q", type=float, default=None)
    parser.add_argument("--edge_certification_csv", default=None)
    return _merge_args(parser.parse_args())


def _parse_dims(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.replace("[", "").replace("]", "").split(",") if x.strip()]


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 3E GAS Graph Audit Summary",
        "",
        "This audit is reset-free and offline-only. It does not run environment",
        "rollouts and does not claim online success.",
        "",
        "`GAS_style_threshold_graph` is a diagnostic proximity approximation, not",
        "an official GAS graph. kNN/proximity edges are treated as untrusted until",
        "they match Phase 2 support-certified option edges.",
        "",
        f"Highest path coverage: `{summary.get('which_graph_has_highest_path_coverage')}`",
        f"Lowest unsupported edge rate: `{summary.get('which_graph_has_lowest_unsupported_edge_rate')}`",
        f"Most unsupported shortcut reliance: `{summary.get('which_graph_relies_most_on_unsupported_shortcuts')}`",
        f"Support graph reduces path risk: `{summary.get('support_certified_graph_reduces_path_risk')}`",
        f"GAS/proximity overestimates connectivity: `{summary.get('gas_style_or_proximity_graph_overestimates_connectivity')}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    dataset_key = _dataset_key(args.dataset_name)
    out_dir = Path(args.output_dir) / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    option_edges, edge_segments, selected_nodes = load_phase2_artifacts(phase2_run_dir)
    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    observations = np.asarray(dataset["observations"])
    cluster_model = fit_state_clusters(
        observations,
        method=str(args.cluster_method),
        n_clusters=int(args.n_clusters),
        seed=int(args.seed),
        state_dims=_parse_dims(args.state_dims),
    )
    labels = np.asarray(cluster_model["labels"], dtype=np.int64)
    edge_sets = build_audit_edge_sets(
        option_edges,
        selected_nodes,
        observations,
        labels,
        n_clusters=int(args.n_clusters),
        knn_k=int(args.knn_k),
        seed=int(args.seed),
    )
    edge_audit = edge_provenance_audit(edge_sets, option_edges)
    _, pair_compat = compute_edge_compatibility(
        option_edges,
        edge_segments,
        labels=np.empty(0, dtype=np.int64),
        pair_records={},
        H_intra=int(args.compat_H_intra),
    )
    path_queries = pd.read_csv(phase2_run_dir.parent / "path_queries.csv")
    cert = (
        pd.read_csv(Path(args.edge_certification_csv).expanduser())
        if args.edge_certification_csv and Path(args.edge_certification_csv).expanduser().exists()
        else pd.DataFrame()
    )
    path_audit = path_risk_audit(
        edge_sets,
        option_edges,
        path_queries,
        edge_certification=cert,
        pair_compatibility=pair_compat,
        max_queries=int(args.max_queries),
        seed=int(args.seed),
    )
    bottleneck_path = phase2_run_dir.parent / f"bottleneck_scores_H{int(option_edges['max_h'].max())}.csv"
    if not bottleneck_path.exists():
        candidates = sorted(phase2_run_dir.parent.glob("bottleneck_scores_H*.csv"))
        bottleneck_path = candidates[0] if candidates else phase2_run_dir / "selected_nodes.csv"
    bottleneck_df = pd.read_csv(bottleneck_path)
    bottleneck = bottleneck_audit(
        edge_sets,
        option_edges,
        bottleneck_df,
        path_queries,
        top_q=float(args.bottleneck_top_q),
        max_queries=int(args.max_queries),
        seed=int(args.seed),
    )
    summary = audit_summary(edge_audit, path_audit, bottleneck)
    edge_audit.to_csv(out_dir / "graph_edge_audit.csv", index=False)
    path_audit.to_csv(out_dir / "graph_path_audit.csv", index=False)
    bottleneck.to_csv(out_dir / "graph_bottleneck_audit.csv", index=False)
    _write_json(out_dir / "gas_audit_summary.json", {**summary, "config": vars(args)})
    _write_markdown(out_dir / "gas_audit_summary.md", summary)
    print(f"[phase3E] wrote GAS graph audit under {out_dir}")
    print(json.dumps(_json_safe(summary), sort_keys=True))


if __name__ == "__main__":
    main()

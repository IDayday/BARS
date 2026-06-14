#!/usr/bin/env python
"""Run Phase 3E reset-free offline option-edge certification."""

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

from phase1.data import load_ogbench_dataset  # noqa: E402
from phase2.compatibility import compute_edge_compatibility  # noqa: E402
from phase3.evaluation import load_phase2_artifacts  # noqa: E402
from phase3e.edge_proxy_score import (  # noqa: E402
    certification_summary,
    compute_compatibility_proxy,
    compute_edge_distance_proxy,
    compute_proxy_scores,
)
from phase3e.heldout_support import compute_heldout_support, segment_indices_for_episode_split  # noqa: E402
from phase3e.plotting import plot_edge_certification  # noqa: E402
from phase3e.policy_likelihood import (  # noqa: E402
    compute_model_edge_mse,
    edge_policy_support_scores,
    load_cached_edge_mse,
)


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _run_name(path: str | Path) -> str:
    return Path(path).expanduser().name


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
        "output_dir": "results/phase3e",
        "max_transitions": 200000,
        "phase3_run_dir": None,
        "model_path": None,
        "edge_metrics_csv": None,
        "heldout_fraction": 0.2,
        "seed": 0,
        "temperature": 0.05,
        "batch_size": 2048,
        "max_policy_eval_examples": 50000,
        "max_segments_per_edge_ood": 128,
        "compat_H_intra": 10,
        "threshold": 0.35,
        "min_heldout_support_lcb": 0.01,
        "w_support": 0.4,
        "w_policy": 0.3,
        "w_compat": 0.2,
        "w_ood": 0.1,
        "device": None,
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
    parser.add_argument("--phase3_run_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--edge_metrics_csv", default=None)
    parser.add_argument("--heldout_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_policy_eval_examples", type=int, default=None)
    parser.add_argument("--max_segments_per_edge_ood", type=int, default=None)
    parser.add_argument("--compat_H_intra", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_heldout_support_lcb", type=float, default=None)
    parser.add_argument("--w_support", type=float, default=None)
    parser.add_argument("--w_policy", type=float, default=None)
    parser.add_argument("--w_compat", type=float, default=None)
    parser.add_argument("--w_ood", type=float, default=None)
    parser.add_argument("--device", default=None)
    return _merge_args(parser.parse_args())


def _resolve_policy_inputs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    model_path = Path(args.model_path).expanduser() if args.model_path else None
    edge_metrics = Path(args.edge_metrics_csv).expanduser() if args.edge_metrics_csv else None
    if args.phase3_run_dir:
        run_dir = Path(args.phase3_run_dir).expanduser()
        if model_path is None and (run_dir / "model.pt").exists():
            model_path = run_dir / "model.pt"
        if edge_metrics is None and (run_dir / "edge_val_metrics.csv").exists():
            edge_metrics = run_dir / "edge_val_metrics.csv"
    return model_path, edge_metrics


def _policy_scores(args: argparse.Namespace, dataset: dict[str, Any], option_edges: pd.DataFrame, edge_segments: dict[str, np.ndarray], heldout_idx: np.ndarray) -> tuple[pd.DataFrame, str]:
    model_path, edge_metrics = _resolve_policy_inputs(args)
    if model_path is not None and model_path.exists():
        edge_mse = compute_model_edge_mse(
            dataset,
            option_edges,
            edge_segments,
            heldout_idx,
            model_path=model_path,
            batch_size=int(args.batch_size),
            max_examples=int(args.max_policy_eval_examples),
            seed=int(args.seed),
            device=args.device,
        )
        source = str(model_path)
    elif edge_metrics is not None and edge_metrics.exists():
        edge_mse = load_cached_edge_mse(edge_metrics)
        source = str(edge_metrics)
    else:
        edge_mse = pd.DataFrame({"edge_id": option_edges["edge_id"], "edge_action_mse": np.nan, "num_policy_eval_samples": 0})
        source = "missing_model_and_validation_cache"
    return edge_policy_support_scores(edge_mse, temperature=float(args.temperature)), source


def _write_markdown(path: Path, summary: dict[str, Any], cert: pd.DataFrame) -> None:
    lines = [
        "# Phase 3E Offline Edge Certification Summary",
        "",
        "This is a reset-free offline proxy. It does not run environment rollout and",
        "is not equivalent to online option execution success.",
        "",
        "The goal is to rank and filter data-supported option edges using heldout",
        "trajectory support, GCBC action-fitting proxy, simple behavior/OOD risk,",
        "and Phase 2 compatibility context.",
        "",
        f"Edges: `{summary.get('num_edges', 0)}`",
        f"Certified offline edges: `{summary.get('certified_offline_edges', 0)}`",
        f"Certified offline rate: `{summary.get('certified_offline_rate', 0.0):.6g}`",
        f"Mean proxy score: `{summary.get('mean_edge_proxy_score', 0.0):.6g}`",
        "",
        "Offline action MSE and proxy scores are risk signals only. Rollout",
        "validation remains Phase 3C/3F work once an environment is available.",
        "",
    ]
    if not cert.empty:
        top = cert.head(10)
        lines += [
            "## Top Edges By Proxy Score",
            "",
            "| edge_id | src | dst | proxy | heldout_lcb | action_mse | certified |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in top.itertuples(index=False):
            lines.append(
                f"| {int(row.edge_id)} | {int(row.src)} | {int(row.dst)} | "
                f"{float(row.edge_proxy_score):.6g} | {float(row.heldout_support_lcb):.6g} | "
                f"{float(row.edge_action_mse) if pd.notna(row.edge_action_mse) else float('nan'):.6g} | "
                f"{bool(row.certified_offline_binary)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    out_dir = Path(args.output_dir) / _dataset_key(args.dataset_name) / _run_name(phase2_run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    option_edges, edge_segments, _ = load_phase2_artifacts(phase2_run_dir)
    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    heldout_support, split_meta = compute_heldout_support(
        option_edges,
        edge_segments,
        heldout_fraction=float(args.heldout_fraction),
        seed=int(args.seed),
    )
    _, heldout_idx, _, _ = segment_indices_for_episode_split(
        edge_segments,
        heldout_fraction=float(args.heldout_fraction),
        seed=int(args.seed),
    )
    policy_scores, policy_source = _policy_scores(args, dataset, option_edges, edge_segments, heldout_idx)
    _, pair_compat = compute_edge_compatibility(
        option_edges,
        edge_segments,
        labels=np.empty(0, dtype=np.int64),
        pair_records={},
        H_intra=int(args.compat_H_intra),
    )
    compat_proxy = compute_compatibility_proxy(pair_compat, option_edges["edge_id"].to_numpy(dtype=np.int64))
    distance_proxy = compute_edge_distance_proxy(
        dataset,
        edge_segments,
        option_edges["edge_id"].to_numpy(dtype=np.int64),
        max_segments_per_edge=int(args.max_segments_per_edge_ood),
        seed=int(args.seed),
    )
    weights = {
        "support": float(args.w_support),
        "policy": float(args.w_policy),
        "compat": float(args.w_compat),
        "ood": float(args.w_ood),
    }
    cert = compute_proxy_scores(
        option_edges,
        heldout_support,
        policy_scores,
        distance_proxy,
        compat_proxy,
        weights=weights,
        threshold=float(args.threshold),
        min_heldout_support_lcb=float(args.min_heldout_support_lcb),
    )
    config = {**vars(args), "policy_source": policy_source, "heldout_split": split_meta}
    summary = certification_summary(cert, config)
    cert.to_csv(out_dir / "offline_edge_certification.csv", index=False)
    _write_json(out_dir / "offline_edge_certification_summary.json", summary)
    _write_markdown(out_dir / "offline_edge_certification_summary.md", summary, cert)
    plot_edge_certification(cert, out_dir)
    print(f"[phase3E] wrote offline edge certification under {out_dir}")
    print(json.dumps(_json_safe(summary), sort_keys=True))


if __name__ == "__main__":
    main()

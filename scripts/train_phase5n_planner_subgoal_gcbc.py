#!/usr/bin/env python
"""Train BARS planner-aware low-level GCBC without environment rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data import load_ogbench_dataset  # noqa: E402
from phase5n.train_planner_gcbc import train_planner_subgoal_gcbc  # noqa: E402


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _parse_hidden_dims(value: str | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.replace("[", "").replace("]", "").split(",") if x.strip()]


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dims", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--val_examples", type=int, default=None)
    parser.add_argument("--train_examples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--edge_embedding_dim", type=int, default=None)
    parser.add_argument("--num_planner_queries", type=int, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "max_transitions": None,
        "batch_size": 1024,
        "num_steps": 100000,
        "lr": 3e-4,
        "hidden_dims": [512, 512, 512],
        "seed": 0,
        "val_fraction": 0.05,
        "val_examples": 8192,
        "train_examples": None,
        "log_interval": None,
        "edge_embedding_dim": 0,
        "source_probabilities": {
            "final_goal_hindsight": 0.25,
            "support_edge_local": 0.35,
            "planner_first_edge_replay": 0.40,
        },
        "source_loss_weights": {
            "final_goal_hindsight": 1.0,
            "support_edge_local": 1.0,
            "planner_first_edge_replay": 1.0,
        },
        "num_planner_queries": 5000,
        "base_loss_weight_mode": "support_bottleneck",
        "base_loss_weight_strength": 0.3,
        "planner_usage_strength": 0.35,
        "planner_first_edge_strength": 0.65,
        "min_loss_weight": 0.5,
        "max_loss_weight": 2.5,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    if not merged.get("dataset_name"):
        raise ValueError("--dataset_name is required")
    if not merged.get("phase2_run_dir"):
        raise ValueError("--phase2_run_dir is required")
    merged["hidden_dims"] = _parse_hidden_dims(merged.get("hidden_dims"))
    if merged.get("output_dir") is None:
        phase2_name = Path(str(merged["phase2_run_dir"])).expanduser().name
        merged["output_dir"] = str(Path("results/phase5n") / _dataset_key(merged["dataset_name"]) / f"{phase2_name}_planner_gcbc")
    return argparse.Namespace(**merged)


def main() -> None:
    args = parse_args()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    train_planner_subgoal_gcbc(
        dataset=dataset,
        option_edges_csv=phase2_run_dir / "option_edges.csv",
        edge_segments_npz=phase2_run_dir / "edge_segments.npz",
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        seed=args.seed,
        val_fraction=args.val_fraction,
        val_examples=args.val_examples,
        train_examples=args.train_examples,
        log_interval=args.log_interval,
        device=args.device,
        edge_embedding_dim=args.edge_embedding_dim,
        source_probabilities=args.source_probabilities,
        source_loss_weights=args.source_loss_weights,
        num_planner_queries=args.num_planner_queries,
        base_loss_weight_mode=args.base_loss_weight_mode,
        base_loss_weight_strength=args.base_loss_weight_strength,
        planner_usage_strength=args.planner_usage_strength,
        planner_first_edge_strength=args.planner_first_edge_strength,
        min_loss_weight=args.min_loss_weight,
        max_loss_weight=args.max_loss_weight,
        config=vars(args),
    )
    print(f"[phase5n] wrote planner-aware GCBC outputs under {args.output_dir}")


if __name__ == "__main__":
    main()


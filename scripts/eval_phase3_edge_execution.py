#!/usr/bin/env python
"""Evaluate GCBC closed-loop executability for Phase 2 and baseline edges."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.clustering import fit_state_clusters  # noqa: E402
from phase1.data import load_ogbench_dataset  # noqa: E402
from phase3.evaluation import (  # noqa: E402
    build_baseline_edge_sets,
    default_phase3_output_dir,
    evaluate_baseline_edge_sets,
    load_phase2_artifacts,
)
from phase3.edge_rollout import evaluate_edge_rollouts  # noqa: E402
from phase3.models import GCBCMLP  # noqa: E402
from phase3.plotting import plot_edge_execution  # noqa: E402
from phase3.reset_utils import probe_reset_capability  # noqa: E402
from phase3.reset_utils import (  # noqa: E402
    RESET_STATUS_ENV_UNAVAILABLE,
    RESET_STATUS_SUPPORTED,
    env_unavailable_probe_result,
    missing_reset_env_packages,
)
from phase3.train_gcbc import write_json  # noqa: E402


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _parse_int_list(value: str | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _infer_H(phase2_run_dir: str | Path, fallback: int = 10) -> int:
    match = re.search(r"_H(\d+)", Path(phase2_run_dir).name)
    return int(match.group(1)) if match else int(fallback)


def _default_cluster_config(dataset_name: str) -> dict[str, Any]:
    if dataset_name.startswith("antmaze"):
        return {"cluster_method": "grid_xy", "n_clusters": 400, "state_dims": [0, 1]}
    if dataset_name.startswith("scene"):
        return {"cluster_method": "kmeans", "n_clusters": 512, "state_dims": None}
    return {"cluster_method": "kmeans", "n_clusters": 512, "state_dims": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--cluster_method", default=None, choices=["kmeans", "grid_xy"])
    parser.add_argument("--state_dims", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--eval_num_edges", type=int, default=None)
    parser.add_argument("--starts_per_edge", type=int, default=None)
    parser.add_argument("--success_mode", default=None, choices=["dst_cluster", "termination_nearest"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--knn_k", type=int, default=None)
    parser.add_argument("--probe_reset_only", action="store_true", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    if not merged.get("dataset_name"):
        raise ValueError("--dataset_name is required")
    cluster_defaults = _default_cluster_config(merged["dataset_name"])
    defaults = {
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "max_transitions": None,
        "eval_num_edges": 100,
        "starts_per_edge": 10,
        "success_mode": "dst_cluster",
        "seed": 0,
        "knn_k": 10,
        "probe_reset_only": False,
        **cluster_defaults,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    if not merged.get("phase2_run_dir"):
        raise ValueError("--phase2_run_dir is required")
    if merged.get("output_dir") is None:
        merged["output_dir"] = str(default_phase3_output_dir(merged["dataset_name"], merged["phase2_run_dir"]))
    if merged.get("model_path") is None:
        merged["model_path"] = str(Path(merged["output_dir"]) / "model.pt")
    if merged.get("H") is None:
        merged["H"] = _infer_H(merged["phase2_run_dir"])
    merged["state_dims"] = _parse_int_list(merged.get("state_dims"))
    return argparse.Namespace(**merged)


def _make_env(dataset_name: str, dataset_dir: str | None) -> Any:
    try:
        import ogbench  # type: ignore

        env, _, _ = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_dir=dataset_dir,
            compact_dataset=False,
        )
        return env
    except Exception as first_exc:
        try:
            import gymnasium as gym

            return gym.make(dataset_name)
        except Exception:
            try:
                import gym

                return gym.make(dataset_name)
            except Exception as second_exc:
                raise RuntimeError(
                    f"Could not construct env {dataset_name!r}: "
                    f"ogbench={type(first_exc).__name__}: {first_exc}; "
                    f"gym={type(second_exc).__name__}: {second_exc}"
                ) from second_exc


def _load_policy(model_path: str | Path, device: torch.device) -> GCBCMLP:
    checkpoint = torch.load(Path(model_path).expanduser(), map_location=device)
    model_config = checkpoint["model_config"]
    model = GCBCMLP(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _dataset_state_ref(dataset: dict[str, Any], index: int) -> dict[str, Any]:
    ref: dict[str, Any] = {"observation": np.asarray(dataset["observations"])[int(index)]}
    for key in ("qpos", "qvel", "state", "states", "sim_state", "sim_states", "infos/qpos", "infos/qvel"):
        value = dataset.get(key)
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] > int(index):
            ref[key] = value[int(index)]
    return ref


def _write_rollout_skip(out: Path, args: argparse.Namespace, probe: dict[str, Any], reason: str) -> None:
    write_json(
        out / "edge_execution_summary.json",
        {
            "dataset_name": args.dataset_name,
            "phase2_run_dir": str(args.phase2_run_dir),
            "output_dir": str(out),
            "rollout_skipped": True,
            "skipped_reason": reason,
            "skip_reason": reason,
            "reset_probe": probe,
            "offline_supervised_metrics_only": True,
        },
    )
    pd.DataFrame(
        [
            {
                "edge_source": "support_option_edges",
                "unsupported_edge_rate": 0.0,
                "num_eval_edges": 0,
                "mean_success_rate": np.nan,
                "median_success_rate": np.nan,
                "skipped": True,
                "skipped_reason": reason,
                "skip_reason": reason,
            }
        ]
    ).to_csv(out / "baseline_edge_execution.csv", index=False)


def _console_json(payload: dict[str, Any]) -> str:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    return json.dumps(safe(payload), sort_keys=True)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    observations = np.asarray(dataset["observations"])
    flat_observations = observations.reshape(observations.shape[0], -1)
    option_edges, edge_segments, selected_nodes = load_phase2_artifacts(args.phase2_run_dir)
    try:
        env = _make_env(args.dataset_name, args.dataset_dir)
    except Exception as exc:
        probe = env_unavailable_probe_result(
            f"env_construction_failed: {type(exc).__name__}: {exc}",
            missing_packages=missing_reset_env_packages(),
            num_probe_states=1,
        )
        write_json(out / "reset_probe.json", {"dataset_name": args.dataset_name, **probe})
        print(f"[phase3] reset probe: {_console_json(probe)}")
        if args.probe_reset_only:
            return
        _write_rollout_skip(out, args, probe, RESET_STATUS_ENV_UNAVAILABLE)
        print("[phase3] rollout skipped; offline supervised metrics remain available")
        return

    probe = probe_reset_capability(env, _dataset_state_ref(dataset, 0))
    write_json(out / "reset_probe.json", {"dataset_name": args.dataset_name, **probe})
    print(f"[phase3] reset probe: {_console_json(probe)}")
    if args.probe_reset_only:
        return
    if probe.get("reset_probe_status") == RESET_STATUS_ENV_UNAVAILABLE:
        _write_rollout_skip(out, args, probe, RESET_STATUS_ENV_UNAVAILABLE)
        print("[phase3] rollout skipped; offline supervised metrics remain available")
        return
    if probe.get("reset_probe_status") != RESET_STATUS_SUPPORTED:
        _write_rollout_skip(out, args, probe, str(probe.get("reset_probe_status", "reset_probe_failed")))
        print("[phase3] rollout skipped; offline supervised metrics remain available")
        return

    cluster_model = fit_state_clusters(
        flat_observations,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
        seed=args.seed,
        state_dims=args.state_dims,
    )
    labels = np.asarray(cluster_model["labels"], dtype=np.int64)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = _load_policy(args.model_path, device)

    support_metrics, support_summary = evaluate_edge_rollouts(
        env,
        policy,
        dataset,
        option_edges,
        edge_segments,
        cluster_model,
        num_edges=args.eval_num_edges,
        starts_per_edge=args.starts_per_edge,
        horizon_mode="edge_median",
        success_mode=args.success_mode,
        seed=args.seed,
        device=device,
    )
    support_metrics.to_csv(out / "edge_rollout_metrics.csv", index=False)
    support_summary.to_csv(out / "edge_rollout_summary.csv", index=False)

    baseline_edges = build_baseline_edge_sets(
        option_edges,
        selected_nodes,
        flat_observations,
        labels,
        n_clusters=int(cluster_model["metadata"]["n_clusters"]),
        edge_budget=int(option_edges.shape[0]),
        knn_k=args.knn_k,
        seed=args.seed,
    )
    all_edge_metrics, baseline_summary = evaluate_baseline_edge_sets(
        env,
        policy,
        dataset,
        baseline_edges,
        option_edges,
        edge_segments,
        labels,
        cluster_model,
        starts_per_edge=args.starts_per_edge,
        H=args.H,
        num_edges=args.eval_num_edges,
        success_mode=args.success_mode,
        seed=args.seed,
    )
    all_edge_metrics.to_csv(out / "all_edge_rollout_metrics.csv", index=False)
    baseline_summary.to_csv(out / "baseline_edge_execution.csv", index=False)
    plot_edge_execution(baseline_summary, out)
    write_json(
        out / "edge_execution_summary.json",
        {
            "dataset_name": args.dataset_name,
            "phase2_run_dir": str(args.phase2_run_dir),
            "output_dir": str(out),
            "support_summary": support_summary.to_dict("records"),
            "baseline_summary": baseline_summary.to_dict("records"),
        },
    )
    print(f"[phase3] wrote edge execution outputs under {out}")


if __name__ == "__main__":
    main()

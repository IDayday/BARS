#!/usr/bin/env python
"""Run reset-free natural-start closed-loop rollout smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3f.natural_rollout import (  # noqa: E402
    load_gcbc_policy,
    resolve_device,
    run_natural_start_episodes,
    write_natural_rollout_outputs,
)
from phase3f.hierarchical_rollout import (  # noqa: E402
    fit_runtime_cluster_model,
    load_graph_artifacts,
    run_hierarchical_support_episodes,
)
from phase3f.task_eval import load_preflight_status, write_env_unavailable_skip  # noqa: E402
from phase1.data import load_ogbench_dataset  # noqa: E402


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _parse_list(value: Any, cast=int) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [cast(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--preflight_root", default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--task_ids", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--action_mode", choices=["direct_gcbc", "random", "zero", "hierarchical_support"], default=None)
    parser.add_argument("--trace_every", type=int, default=None)
    parser.add_argument("--keep_going_after_success", action="store_true")
    parser.add_argument("--skip_preflight_check", action="store_true")
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--cluster_method", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--state_dims", default=None)
    parser.add_argument("--graph_edges_csv", default=None)
    parser.add_argument("--graph_segments_npz", default=None)
    parser.add_argument("--bank_edges_csv", default=None)
    parser.add_argument("--bank_segments_npz", default=None)
    parser.add_argument("--disable_bank_connectors", action="store_true")
    parser.add_argument("--allow_full_bank_fallback", action="store_true")
    parser.add_argument("--edge_horizon_multiplier", type=float, default=None)
    parser.add_argument("--max_edge_horizon", type=int, default=None)
    parser.add_argument("--max_replans", type=int, default=None)
    parser.add_argument("--subgoal_max_candidates", type=int, default=None)
    return parser.parse_args()


def merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None or merged.get(key) is False:
            merged[key] = value
    defaults = {
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "method": "direct_gcbc_natural_start",
        "output_dir": "results/phase3f",
        "preflight_root": "results/phase3/env_preflight_gcrlo",
        "num_episodes": 2,
        "max_steps": 100,
        "task_ids": [],
        "seed": 0,
        "device": "auto",
        "action_mode": "direct_gcbc",
        "trace_every": 1,
        "skip_preflight_check": False,
        "keep_going_after_success": False,
        "max_transitions": 200000,
        "cluster_method": None,
        "n_clusters": None,
        "state_dims": None,
        "graph_edges_csv": None,
        "graph_segments_npz": None,
        "bank_edges_csv": None,
        "bank_segments_npz": None,
        "disable_bank_connectors": False,
        "allow_full_bank_fallback": False,
        "edge_horizon_multiplier": 2.0,
        "max_edge_horizon": None,
        "max_replans": 5,
        "subgoal_max_candidates": 256,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    if not merged.get("dataset_name"):
        raise ValueError("--dataset_name is required")
    merged["task_ids"] = _parse_list(merged.get("task_ids"), int)
    if merged.get("state_dims") is not None:
        merged["state_dims"] = _parse_list(merged.get("state_dims"), int)
    return argparse.Namespace(**merged)


def _construct_env(dataset_name: str, dataset_dir: str):
    import ogbench

    return ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=dataset_dir,
        env_only=True,
    )


def _write_config(path: Path, args: argparse.Namespace, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload.update(extra)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_json_safe(payload), f, sort_keys=True)


def main() -> None:
    args = merge_args(parse_args())
    dataset_key = _dataset_key(args.dataset_name)
    out_dir = Path(args.output_dir) / dataset_key / args.method
    if not args.skip_preflight_check:
        preflight = load_preflight_status(args.dataset_name, args.preflight_root)
        if preflight.get("status") == "env_unavailable" or not preflight.get("ogbench_env_constructed", False):
            reason = str(preflight.get("status", preflight.get("failure_reason", "env_unavailable")))
            write_env_unavailable_skip(out_dir, args.dataset_name, args.method, reason)
            _write_config(out_dir / "config_resolved.yaml", args, {"preflight_status": preflight})
            print(f"[phase3F] skipped natural-start rollout: {reason} output_dir={out_dir}")
            return

    device = resolve_device(args.device)
    policy = None
    if args.action_mode in {"direct_gcbc", "hierarchical_support"}:
        if not args.model_path:
            write_env_unavailable_skip(out_dir, args.dataset_name, args.method, "missing_model_path")
            _write_config(out_dir / "config_resolved.yaml", args, {"device_resolved": str(device)})
            print(f"[phase3F] skipped natural-start rollout: missing_model_path output_dir={out_dir}")
            return
        if not Path(args.model_path).expanduser().exists():
            write_env_unavailable_skip(out_dir, args.dataset_name, args.method, f"model_missing:{args.model_path}")
            _write_config(out_dir / "config_resolved.yaml", args, {"device_resolved": str(device)})
            print(f"[phase3F] skipped natural-start rollout: model_missing output_dir={out_dir}")
            return
        policy = load_gcbc_policy(args.model_path, device)

    try:
        env = _construct_env(args.dataset_name, args.dataset_dir)
    except Exception as exc:
        reason = f"env_construction_failed:{type(exc).__name__}: {exc}"
        write_env_unavailable_skip(out_dir, args.dataset_name, args.method, reason)
        _write_config(
            out_dir / "config_resolved.yaml",
            args,
            {"device_resolved": str(device), "env_error_trace": traceback.format_exc()},
        )
        print(f"[phase3F] skipped natural-start rollout: {reason} output_dir={out_dir}")
        return

    if args.action_mode == "hierarchical_support":
        if not args.cluster_method or args.n_clusters is None:
            write_env_unavailable_skip(out_dir, args.dataset_name, args.method, "missing_cluster_config")
            _write_config(out_dir / "config_resolved.yaml", args, {"device_resolved": str(device)})
            print(f"[phase3F] skipped hierarchical rollout: missing_cluster_config output_dir={out_dir}")
            return
        if not args.graph_edges_csv or not args.graph_segments_npz:
            write_env_unavailable_skip(out_dir, args.dataset_name, args.method, "missing_graph_artifacts")
            _write_config(out_dir / "config_resolved.yaml", args, {"device_resolved": str(device)})
            print(f"[phase3F] skipped hierarchical rollout: missing_graph_artifacts output_dir={out_dir}")
            return
        dataset = load_ogbench_dataset(
            args.dataset_name,
            args.dataset_dir,
            split="train",
            max_transitions=args.max_transitions,
        )
        cluster_model = fit_runtime_cluster_model(
            dataset,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            seed=args.seed,
            state_dims=args.state_dims,
        )
        graph_edges, graph_segments, bank_edges, bank_segments = load_graph_artifacts(
            graph_edges_csv=args.graph_edges_csv,
            graph_segments_npz=args.graph_segments_npz,
            bank_edges_csv=args.bank_edges_csv,
            bank_segments_npz=args.bank_segments_npz,
        )
        episodes, traces = run_hierarchical_support_episodes(
            env,
            policy,
            dataset=dataset,
            cluster_model=cluster_model,
            graph_edges=graph_edges,
            graph_segments=graph_segments,
            bank_edges=bank_edges,
            bank_segments=bank_segments,
            dataset_name=args.dataset_name,
            method=args.method,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            task_ids=args.task_ids,
            seed=args.seed,
            device=str(device),
            allow_bank_connectors=not args.disable_bank_connectors,
            edge_horizon_multiplier=args.edge_horizon_multiplier,
            max_edge_horizon=args.max_edge_horizon,
            max_replans=args.max_replans,
            subgoal_max_candidates=args.subgoal_max_candidates,
            allow_full_bank_fallback=args.allow_full_bank_fallback,
        )
    else:
        episodes, traces = run_natural_start_episodes(
            env,
            policy,
            dataset_name=args.dataset_name,
            method=args.method,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            task_ids=args.task_ids,
            seed=args.seed,
            action_mode=args.action_mode,
            device=device,
            stop_on_success=not args.keep_going_after_success,
            trace_every=args.trace_every,
        )
    summary = write_natural_rollout_outputs(
        out_dir,
        dataset_name=args.dataset_name,
        method=args.method,
        episodes=episodes,
        traces=traces,
        skipped=False,
        skipped_reason="",
    )
    _write_config(out_dir / "config_resolved.yaml", args, {"device_resolved": str(device)})
    (out_dir / "natural_rollout_summary.json").write_text(
        json.dumps(summary.iloc[0].to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[phase3F] wrote natural-start rollout outputs under {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

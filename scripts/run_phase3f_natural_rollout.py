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
    load_or_fit_runtime_cluster_model,
    load_graph_artifacts,
    run_hierarchical_support_episodes,
)
from phase3f.edge_memory import (  # noqa: E402
    load_edge_memory,
    memory_failed_edge_counts,
    write_edge_memory_outputs,
)
from phase3f.edge_outcome_model import (  # noqa: E402
    edge_outcome_penalty_map,
    fit_edge_outcome_scores,
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
    parser.add_argument("--cluster_cache_dir", default=None)
    parser.add_argument("--cluster_cache_path", default=None)
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
    parser.add_argument("--failure_penalty", type=float, default=None)
    parser.add_argument("--initiation_weight", type=float, default=None)
    parser.add_argument("--downstream_weight", type=float, default=None)
    parser.add_argument("--policy_mse_weight", type=float, default=None)
    parser.add_argument("--policy_mse_scale", type=float, default=None)
    parser.add_argument("--edge_memory_csv", default=None)
    parser.add_argument("--use_edge_memory", action="store_true")
    parser.add_argument("--update_edge_memory", action="store_true")
    parser.add_argument("--memory_penalty_mode", default=None)
    parser.add_argument("--use_edge_outcome_model", action="store_true")
    parser.add_argument("--outcome_penalty_weight", type=float, default=None)
    parser.add_argument("--outcome_min_attempts", type=int, default=None)
    parser.add_argument("--outcome_alpha", type=float, default=None)
    parser.add_argument("--outcome_beta", type=float, default=None)
    parser.add_argument("--outcome_uncertainty_weight", type=float, default=None)
    parser.add_argument("--outcome_policy_mse_weight", type=float, default=None)
    parser.add_argument("--outcome_policy_mse_scale", type=float, default=None)
    parser.add_argument("--outcome_subgoal_l2_weight", type=float, default=None)
    parser.add_argument("--outcome_subgoal_l2_scale", default=None)
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
        "cluster_cache_dir": None,
        "cluster_cache_path": None,
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
        "failure_penalty": 0.0,
        "initiation_weight": 1.0,
        "downstream_weight": 0.25,
        "policy_mse_weight": 0.0,
        "policy_mse_scale": 0.05,
        "edge_memory_csv": None,
        "use_edge_memory": False,
        "update_edge_memory": False,
        "memory_penalty_mode": "failure_excess",
        "use_edge_outcome_model": False,
        "outcome_penalty_weight": 1.0,
        "outcome_min_attempts": 1,
        "outcome_alpha": 1.0,
        "outcome_beta": 1.0,
        "outcome_uncertainty_weight": 0.25,
        "outcome_policy_mse_weight": 0.0,
        "outcome_policy_mse_scale": 0.05,
        "outcome_subgoal_l2_weight": 0.0,
        "outcome_subgoal_l2_scale": "auto",
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


def _cluster_cache_path(args: argparse.Namespace, dataset_key: str) -> Path | None:
    if args.cluster_cache_path:
        return Path(args.cluster_cache_path)
    if not args.cluster_cache_dir:
        return None
    dims = "all" if args.state_dims is None else "-".join(str(x) for x in args.state_dims)
    name = (
        f"{dataset_key}_{args.cluster_method}_k{int(args.n_clusters)}_"
        f"seed{int(args.seed)}_max{int(args.max_transitions or 0)}_dims{dims}.pkl"
    )
    return Path(args.cluster_cache_dir) / name


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
        cache_path = _cluster_cache_path(args, dataset_key)
        cluster_model, cluster_cache_hit = load_or_fit_runtime_cluster_model(
            dataset,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            seed=args.seed,
            state_dims=args.state_dims,
            cache_path=cache_path,
        )
        graph_edges, graph_segments, bank_edges, bank_segments = load_graph_artifacts(
            graph_edges_csv=args.graph_edges_csv,
            graph_segments_npz=args.graph_segments_npz,
            bank_edges_csv=args.bank_edges_csv,
            bank_segments_npz=args.bank_segments_npz,
        )
        needs_memory = bool(args.use_edge_memory or args.use_edge_outcome_model)
        edge_memory = load_edge_memory(args.edge_memory_csv) if needs_memory else load_edge_memory(None)
        prior_failed_counts = (
            memory_failed_edge_counts(edge_memory, mode=args.memory_penalty_mode)
            if args.use_edge_memory
            else {}
        )
        outcome_scores = fit_edge_outcome_scores(
            edge_memory,
            alpha=args.outcome_alpha,
            beta=args.outcome_beta,
            min_attempts=args.outcome_min_attempts,
            penalty_weight=args.outcome_penalty_weight,
            uncertainty_weight=args.outcome_uncertainty_weight,
            policy_mse_weight=args.outcome_policy_mse_weight,
            policy_mse_scale=args.outcome_policy_mse_scale,
            subgoal_l2_weight=args.outcome_subgoal_l2_weight,
            subgoal_l2_scale=args.outcome_subgoal_l2_scale,
        ) if args.use_edge_outcome_model else fit_edge_outcome_scores(None)
        edge_risk_penalties = edge_outcome_penalty_map(outcome_scores)
        if args.use_edge_outcome_model:
            out_dir.mkdir(parents=True, exist_ok=True)
            outcome_scores.to_csv(out_dir / "edge_outcome_scores.csv", index=False)
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
            failure_penalty=args.failure_penalty,
            initiation_weight=args.initiation_weight,
            downstream_weight=args.downstream_weight,
            policy_mse_weight=args.policy_mse_weight,
            policy_mse_scale=args.policy_mse_scale,
            prior_failed_edge_counts=prior_failed_counts,
            edge_risk_penalties=edge_risk_penalties,
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
    if args.action_mode == "hierarchical_support":
        write_edge_memory_outputs(
            out_dir,
            traces,
            edge_memory_csv=args.edge_memory_csv,
            update_memory=args.update_edge_memory,
            memory_penalty_mode=args.memory_penalty_mode,
        )
    extra = {"device_resolved": str(device)}
    if args.action_mode == "hierarchical_support":
        extra.update(
            {
                "cluster_cache_path_resolved": str(cache_path) if cache_path is not None else None,
                "cluster_cache_hit": bool(cluster_cache_hit),
                "edge_memory_csv_resolved": args.edge_memory_csv,
                "edge_memory_used": bool(args.use_edge_memory),
                "edge_memory_updated": bool(args.update_edge_memory),
                "edge_memory_prior_penalized_edges": int(len(prior_failed_counts)),
                "edge_outcome_model_used": bool(args.use_edge_outcome_model),
                "edge_outcome_num_scored_edges": int(outcome_scores.shape[0]),
                "edge_outcome_num_penalized_edges": int(len(edge_risk_penalties)),
            }
        )
    _write_config(out_dir / "config_resolved.yaml", args, extra)
    (out_dir / "natural_rollout_summary.json").write_text(
        json.dumps(summary.iloc[0].to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[phase3F] wrote natural-start rollout outputs under {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

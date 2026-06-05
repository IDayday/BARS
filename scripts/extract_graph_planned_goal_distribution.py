#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from cage_gp0_common import (
    DEFAULT_GP0_ENVS,
    embeddings_path,
    gas_initial_target_index,
    keygraph_path,
    load_keygraph,
    manifest_path,
    path_position_bucket,
    read_json,
    select_task_ids,
    shortest_path_indices,
    summarize_numeric,
    write_json,
    write_jsonl,
)

import sys

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.state_ref import (  # noqa: E402
    EXACT_MUJOCO_STATE,
    OBSERVATION_ONLY_NOT_EXACT,
    serialize_state_ref,
    make_state_ref_from_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GAS graph-planned state-goal distribution q_G without executing the policy.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_GP0_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--task_ids", nargs="*", type=int, default=None)
    parser.add_argument("--num_start_samples", type=int, default=2000)
    parser.add_argument("--max_paths_per_task", type=int, default=500)
    parser.add_argument("--eval_subgoal_threshold", type=float, default=None)
    parser.add_argument("--eval_final_goal_threshold", type=int, default=2)
    parser.add_argument("--audit_seed", type=int, default=0)
    parser.add_argument("--include_vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store_state_refs", action="store_true")
    parser.add_argument("--state_ref_mode", choices=["exact_only", "best_effort", "metadata_only"], default="metadata_only")
    parser.add_argument("--max_pairs_per_env", type=int, default=0)
    parser.add_argument("--include_qpos_qvel_if_available", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_jsonl", default=None)
    parser.add_argument("--out_summary_json", default=None)
    return parser.parse_args()


def load_dataset_observations(env_name: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not env_name.startswith(("antmaze", "humanoidmaze")):
        return None, {"dataset_load_error": "only OGBench locomaze envs are supported for state refs"}
    try:
        import ogbench  # type: ignore
        env, train_dataset, _ = ogbench.make_env_and_datasets(env_name, compact_dataset=False)
        obs, _ = env.reset(seed=0)
        unwrapped = getattr(env, "unwrapped", env)
        qpos_shape = None
        qvel_shape = None
        obs_equals_qpos_qvel = False
        if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos") and hasattr(unwrapped.data, "qvel"):
            qpos_shape = tuple(np.asarray(unwrapped.data.qpos).shape)
            qvel_shape = tuple(np.asarray(unwrapped.data.qvel).shape)
            concat = np.concatenate([np.asarray(unwrapped.data.qpos).ravel(), np.asarray(unwrapped.data.qvel).ravel()])
            obs_arr = np.asarray(obs).ravel()
            obs_equals_qpos_qvel = bool(obs_arr.shape == concat.shape and np.allclose(obs_arr, concat))
        return np.asarray(train_dataset["observations"]), {
            "dataset_load_error": None,
            "qpos_shape": qpos_shape,
            "qvel_shape": qvel_shape,
            "obs_equals_qpos_qvel": obs_equals_qpos_qvel,
        }
    except Exception as exc:
        return None, {"dataset_load_error": f"{type(exc).__name__}: {exc}"}


def state_ref_for_dataset_idx(
    env_name: str,
    seed: int,
    dataset_obs: np.ndarray | None,
    dataset_idx: int,
    phi: np.ndarray,
    state_ref_meta: dict[str, Any],
    include_qpos_qvel: bool,
    mode: str,
) -> tuple[dict[str, Any] | None, bool, str]:
    if dataset_obs is None:
        return None, False, state_ref_meta.get("dataset_load_error") or "dataset observations unavailable"
    if dataset_idx < 0 or dataset_idx >= len(dataset_obs):
        return None, False, "dataset index out of range"
    obs = np.asarray(dataset_obs[dataset_idx], dtype=np.float32)
    qpos = None
    qvel = None
    reset_mode = OBSERVATION_ONLY_NOT_EXACT
    if include_qpos_qvel and state_ref_meta.get("obs_equals_qpos_qvel"):
        qpos_len = int(np.prod(state_ref_meta.get("qpos_shape") or (0,)))
        qvel_len = int(np.prod(state_ref_meta.get("qvel_shape") or (0,)))
        if qpos_len + qvel_len == obs.size and qpos_len > 0 and qvel_len > 0:
            qpos = obs[:qpos_len].reshape(state_ref_meta["qpos_shape"])
            qvel = obs[qpos_len : qpos_len + qvel_len].reshape(state_ref_meta["qvel_shape"])
            reset_mode = EXACT_MUJOCO_STATE
    exact = reset_mode == EXACT_MUJOCO_STATE
    if mode == "exact_only" and not exact:
        return None, False, "exact StateRef unavailable"
    ref = make_state_ref_from_observation(
        env_name,
        obs,
        phi=phi,
        seed=seed,
        dataset_name=env_name,
        source="dataset",
        qpos=qpos,
        qvel=qvel,
        reset_mode=reset_mode,
        metadata={"dataset_idx": int(dataset_idx)},
    )
    return serialize_state_ref(ref), exact, "exact" if exact else "observation only is not exact"


def planned_rows_for_env_seed(args: argparse.Namespace, env_name: str, seed: int) -> Iterator[dict]:
    kg_path = keygraph_path(args.checkpoint_root, env_name, seed)
    emb_path = embeddings_path(args.checkpoint_root, env_name, seed)
    if not kg_path.exists():
        raise FileNotFoundError(f"Missing keygraph: {kg_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing dataset embeddings: {emb_path}")

    keygraph = load_keygraph(kg_path)
    embeddings = np.load(emb_path, mmap_mode="r")
    nodes = np.asarray(keygraph["nodes"], dtype=np.float32)
    task_ids = select_task_ids(keygraph, args.task_ids)
    subgoal_threshold = float(args.eval_subgoal_threshold or keygraph["way_steps"])
    final_goal_threshold = int(args.eval_final_goal_threshold)
    manifest = read_json(manifest_path(args.checkpoint_root, env_name, seed), default={}) or {}
    rng = np.random.default_rng(args.audit_seed + int(seed))
    num_samples = min(int(args.num_start_samples), len(embeddings))
    start_indices = rng.choice(len(embeddings), size=num_samples, replace=False)
    dataset_obs = None
    state_ref_meta: dict[str, Any] = {}
    if args.store_state_refs:
        dataset_obs, state_ref_meta = load_dataset_observations(env_name)

    per_task_counts = {task_id: 0 for task_id in task_ids}
    path_counter = 0
    emitted_pairs = 0
    for start_idx in start_indices:
        start_phi = np.asarray(embeddings[int(start_idx)], dtype=np.float32)
        for task_id in task_ids:
            if args.max_pairs_per_env and emitted_pairs >= args.max_pairs_per_env:
                return
            if per_task_counts[task_id] >= args.max_paths_per_task:
                continue
            path_nodes, graph_distance = shortest_path_indices(keygraph, task_id, start_phi, force_closest=True)
            if not path_nodes:
                continue
            per_task_counts[task_id] += 1
            path_id = f"{env_name}__seed{seed}__task{task_id}__path{path_counter}"
            path_counter += 1
            path_length = len(path_nodes)
            initial_target_pos = gas_initial_target_index(path_nodes, nodes, start_phi, subgoal_threshold)
            target_node = path_nodes[initial_target_pos]
            target_phi = nodes[target_node]
            d_phi = float(np.linalg.norm(target_phi - start_phi))
            final_phase = path_length <= final_goal_threshold or initial_target_pos >= max(0, path_length - final_goal_threshold)
            row = {
                "record_type": "graph_pair",
                "env_name": env_name,
                "seed": int(seed),
                "path_idx": path_id,
                "task_id": int(task_id),
                "start_dataset_idx": int(start_idx),
                "path_position": int(initial_target_pos),
                "path_length": int(path_length),
                "path_position_bucket": path_position_bucket(initial_target_pos, path_length, final_phase),
                "pair_role": "initial_planner_target",
                "node_idx": None,
                "target_node_idx": int(target_node),
                "d_phi": d_phi,
                "edge_length": d_phi,
                "graph_distance_to_goal": None if graph_distance is None else float(graph_distance),
                "final_phase": bool(final_phase),
                "eval_subgoal_threshold": subgoal_threshold,
                "eval_final_goal_threshold": final_goal_threshold,
                "way_steps": float(keygraph["way_steps"]),
                "source": "gas_graph_planner",
                "checkpoint_root": str(args.checkpoint_root),
                "keygraph_path": str(kg_path),
                "embeddings_path": str(emb_path),
                "manifest_source": manifest.get("source"),
            }
            if args.include_vectors:
                row["s_ref"] = start_phi
                row["g_ref"] = target_phi
                row["phi_s"] = start_phi
                row["phi_g"] = target_phi
            add_state_ref_fields(row, args, env_name, seed, dataset_obs, int(start_idx), start_phi, None, target_phi, state_ref_meta)
            emitted_pairs += 1
            yield row

            for pos in range(max(0, path_length - 1)):
                if args.max_pairs_per_env and emitted_pairs >= args.max_pairs_per_env:
                    return
                s_node = path_nodes[pos]
                g_node = path_nodes[pos + 1]
                s_ref = nodes[s_node]
                g_ref = nodes[g_node]
                d_phi = float(np.linalg.norm(g_ref - s_ref))
                final_phase = (path_length - pos) <= final_goal_threshold or pos >= max(0, path_length - final_goal_threshold)
                row = {
                    "record_type": "graph_pair",
                    "env_name": env_name,
                    "seed": int(seed),
                    "path_idx": path_id,
                    "task_id": int(task_id),
                    "start_dataset_idx": int(start_idx),
                    "path_position": int(pos),
                    "path_length": int(path_length),
                    "path_position_bucket": path_position_bucket(pos, path_length, final_phase),
                    "pair_role": "path_edge",
                    "node_idx": int(s_node),
                    "target_node_idx": int(g_node),
                    "d_phi": d_phi,
                    "edge_length": d_phi,
                    "graph_distance_to_goal": None,
                    "final_phase": bool(final_phase),
                    "eval_subgoal_threshold": subgoal_threshold,
                    "eval_final_goal_threshold": final_goal_threshold,
                    "way_steps": float(keygraph["way_steps"]),
                    "source": "gas_graph_planner",
                    "checkpoint_root": str(args.checkpoint_root),
                    "keygraph_path": str(kg_path),
                    "embeddings_path": str(emb_path),
                    "manifest_source": manifest.get("source"),
                }
                if args.include_vectors:
                    row["s_ref"] = s_ref
                    row["g_ref"] = g_ref
                    row["phi_s"] = s_ref
                    row["phi_g"] = g_ref
                add_state_ref_fields(row, args, env_name, seed, dataset_obs, -1, s_ref, int(s_node), g_ref, state_ref_meta)
                emitted_pairs += 1
                yield row


def add_state_ref_fields(
    row: dict[str, Any],
    args: argparse.Namespace,
    env_name: str,
    seed: int,
    dataset_obs: np.ndarray | None,
    dataset_idx: int,
    phi_s: np.ndarray,
    node_idx: int | None,
    phi_g: np.ndarray,
    state_ref_meta: dict[str, Any],
) -> None:
    row["edge_type"] = row.get("pair_role")
    row["phi_s"] = phi_s
    row["phi_g"] = phi_g
    row["s_ref_exact_reset_capable"] = False
    row["g_ref_exact_reset_capable"] = False
    row["probeable"] = False
    row["non_probeable_reason"] = "state refs not requested"
    if not args.store_state_refs:
        return
    row["state_ref_mode"] = args.state_ref_mode
    row["state_ref_dataset_meta"] = {
        k: v for k, v in state_ref_meta.items() if k in {"dataset_load_error", "qpos_shape", "qvel_shape", "obs_equals_qpos_qvel"}
    }
    if dataset_idx >= 0:
        state_ref_s, exact_s, reason_s = state_ref_for_dataset_idx(
            env_name,
            seed,
            dataset_obs,
            dataset_idx,
            phi_s,
            state_ref_meta,
            args.include_qpos_qvel_if_available,
            args.state_ref_mode,
        )
        if state_ref_s is not None:
            row["state_ref_s"] = state_ref_s
        row["s_ref_exact_reset_capable"] = exact_s
        row["probeable"] = exact_s
        row["non_probeable_reason"] = None if exact_s else reason_s
    else:
        row["non_probeable_reason"] = "path-edge source is a keygraph phi node, not a raw dataset simulator state"
    row["state_ref_g"] = {
        "env_name": env_name,
        "seed": int(seed),
        "phi": phi_g.tolist(),
        "source": "planner_target_phi",
        "reset_mode": OBSERVATION_ONLY_NOT_EXACT,
        "exact_reset": False,
        "metadata": {"node_idx": node_idx, "note": "goal restore is not required for policy probe; target is phi_g"},
    }


def summarize(rows: list[dict]) -> dict:
    by_env: dict[str, list[dict]] = {}
    for row in rows:
        by_env.setdefault(row["env_name"], []).append(row)
    env_rows = {}
    for env_name, env_records in by_env.items():
        env_rows[env_name] = {
            "num_pairs": len(env_records),
            "num_paths": len({r["path_idx"] for r in env_records}),
            "num_initial_pairs": sum(r.get("pair_role") == "initial_planner_target" for r in env_records),
            "num_path_edges": sum(r.get("pair_role") == "path_edge" for r in env_records),
            "final_phase_rate": float(np.mean([bool(r.get("final_phase")) for r in env_records])) if env_records else None,
            "d_phi": summarize_numeric(r.get("d_phi") for r in env_records),
            "path_length": summarize_numeric(r.get("path_length") for r in env_records if r.get("pair_role") == "initial_planner_target"),
        }
    return {
        "num_pairs": len(rows),
        "envs": sorted(by_env),
        "by_env": env_rows,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else output_root / "graph_planned_pairs.jsonl"
    out_summary = Path(args.out_summary_json) if args.out_summary_json else output_root / "graph_planned_summary.json"

    rows: list[dict] = []
    for env_name in args.envs:
        for seed in args.seeds:
            root = Path(args.checkpoint_root) / env_name / f"seed{seed}"
            if not root.exists():
                continue
            rows.extend(planned_rows_for_env_seed(args, env_name, seed))
    count = write_jsonl(out_jsonl, rows)
    summary = summarize(rows)
    summary.update({"out_jsonl": str(out_jsonl), "count_written": count})
    write_json(out_summary, summary)
    print({"out_jsonl": str(out_jsonl), "out_summary_json": str(out_summary), "rows": count})


if __name__ == "__main__":
    main()

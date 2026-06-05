#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import iter_jsonl, pearson, policy_path, summarize_numeric, write_json, write_jsonl

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.state_ref import deserialize_state_ref, restore_env_from_state_ref, state_ref_is_exact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop probe for frozen GAS low-level policy on graph-planned pairs.")
    parser.add_argument("--mode", choices=["execute", "trace_proxy", "dry_run"], default=None)
    parser.add_argument("--pairs_path", default=None)
    parser.add_argument("--pair_jsonl", default=None, help="Backward-compatible alias for --pairs_path.")
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--env_name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_pairs", type=int, default=128)
    parser.add_argument("--max_pairs", type=int, default=None, help="Backward-compatible alias for --num_pairs.")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out_jsonl", default=None)
    parser.add_argument("--out_summary", default=None)
    parser.add_argument("--out_json", default=None, help="Trace-proxy output JSON.")
    parser.add_argument("--out_md", default=None, help="Trace-proxy output markdown.")
    parser.add_argument("--pair_support_jsonl", default=None)
    parser.add_argument("--trace_roots", nargs="*", default=None)
    parser.add_argument("--allow_approximate_reset", action="store_true")
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if args.pairs_path or args.pair_jsonl:
        return "execute"
    return "trace_proxy"


def load_support(path: str | None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path else []


def find_trace_files(roots: list[str] | None) -> list[Path]:
    files: list[Path] = []
    for root in roots or []:
        p = Path(root)
        if p.is_file() and p.name.endswith(".jsonl"):
            files.append(p)
        elif p.exists():
            files.extend(sorted(p.rglob("cage_trace.jsonl")))
    return sorted(set(files))


def support_by_env_task(support_rows: list[dict[str, Any]]) -> dict[tuple[str, int | None], dict[str, Any]]:
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in support_rows:
        groups.setdefault((str(row.get("env_name")), row.get("task_id")), []).append(row)
    out: dict[tuple[str, int | None], dict[str, Any]] = {}
    for key, rows in groups.items():
        out[key] = {
            "support_score_mean": summarize_numeric(r.get("q_train_support_score") for r in rows)["mean"],
            "coverage_rate": float(np.mean([bool(r.get("q_train_covered")) for r in rows])) if rows else None,
            "graph_d_phi_mean": summarize_numeric(r.get("d_phi") for r in rows)["mean"],
            "num_support_pairs": len(rows),
        }
    env_groups: dict[str, list[dict[str, Any]]] = {}
    for row in support_rows:
        env_groups.setdefault(str(row.get("env_name")), []).append(row)
    for env_name, rows in env_groups.items():
        out[(env_name, None)] = {
            "support_score_mean": summarize_numeric(r.get("q_train_support_score") for r in rows)["mean"],
            "coverage_rate": float(np.mean([bool(r.get("q_train_covered")) for r in rows])) if rows else None,
            "graph_d_phi_mean": summarize_numeric(r.get("d_phi") for r in rows)["mean"],
            "num_support_pairs": len(rows),
        }
    return out


def trace_proxy_probe(args: argparse.Namespace) -> dict[str, Any]:
    support_lookup = support_by_env_task(load_support(args.pair_support_jsonl))
    episodes = []
    for path in find_trace_files(args.trace_roots):
        for row in iter_jsonl(path):
            if row.get("record_type", "episode") == "episode":
                row = dict(row)
                row["trace_path"] = str(path)
                episodes.append(row)
    joined = []
    for row in episodes:
        env_name = str(row.get("env_name"))
        support = support_lookup.get((env_name, row.get("task_id"))) or support_lookup.get((env_name, None)) or {}
        joined.append(
            {
                "env_name": env_name,
                "variant": variant_from_trace_path(row.get("trace_path", "")),
                "success": row.get("success"),
                "segment_target_reach_rate": row.get("segment_target_reach_rate"),
                "support_score_mean": support.get("support_score_mean"),
                "graph_d_phi_mean": support.get("graph_d_phi_mean"),
                "global_replan_request_count": row.get("global_replan_request_count"),
            }
        )
    by_env_variant = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in joined:
        groups.setdefault((row["env_name"], row["variant"]), []).append(row)
    for (env_name, variant), rows in sorted(groups.items()):
        by_env_variant.append(
            {
                "env_name": env_name,
                "variant": variant,
                "num_episodes": len(rows),
                "success_mean": summarize_numeric(r.get("success") for r in rows)["mean"],
                "segment_target_reach_rate_mean": summarize_numeric(r.get("segment_target_reach_rate") for r in rows)["mean"],
                "support_score_mean": summarize_numeric(r.get("support_score_mean") for r in rows)["mean"],
                "graph_d_phi_mean": summarize_numeric(r.get("graph_d_phi_mean") for r in rows)["mean"],
                "global_replan_request_count_mean": summarize_numeric(r.get("global_replan_request_count") for r in rows)["mean"],
            }
        )
    return {
        "mode": "trace_proxy",
        "num_joined_episodes": len(joined),
        "correlations": {
            "segment_reach_vs_support": pearson([r.get("segment_target_reach_rate") for r in joined], [r.get("support_score_mean") for r in joined]),
            "segment_reach_vs_graph_d_phi": pearson([r.get("segment_target_reach_rate") for r in joined], [r.get("graph_d_phi_mean") for r in joined]),
            "success_vs_support": pearson([r.get("success") for r in joined], [r.get("support_score_mean") for r in joined]),
            "success_vs_graph_d_phi": pearson([r.get("success") for r in joined], [r.get("graph_d_phi_mean") for r in joined]),
        },
        "by_env_variant": by_env_variant,
    }


def variant_from_trace_path(path: str) -> str:
    known = {"gas", "cage_trace_only", "cage_fixed_commit", "cage_drift_only", "cage_recovery_only", "cage_full", "cage_safe_full"}
    for part in Path(path).parts:
        if part in known:
            return part
    return "unknown"


def setup_gas_agent(args: argparse.Namespace):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if str(args.gpu) == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import jax
    import numpy as jnp_np

    from M_utils.agents import agents_dict
    from M_utils.agents.gas import get_config
    from M_utils.flax_utils import restore_agent
    from O_utils.datasets import Dataset, GCDataset
    from O_utils.env_utils import make_env_and_datasets

    env, train_dataset, _ = make_env_and_datasets(args.env_name, args.seed)
    config = get_config()
    train_gc_dataset = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = train_gc_dataset.sample(1)
    agent = agents_dict[config["agent_name"]].create(args.seed, example_batch["observations"], example_batch["actions"], config)
    ckpt = Path(policy_path(args.checkpoint_root, args.env_name, args.seed))
    restore_dir = str(ckpt.parent)
    restore_epoch = ckpt.stem.split("_")[-1]
    agent = restore_agent(agent, restore_dir, restore_epoch)
    return env, agent, jax


def current_observation(env: Any) -> np.ndarray:
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "get_ob"):
        return np.asarray(unwrapped.get_ob(), dtype=np.float32)
    if hasattr(unwrapped, "state_vector"):
        return np.asarray(unwrapped.state_vector(), dtype=np.float32)
    raise RuntimeError("Environment does not expose get_ob/state_vector after set_state")


def step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return np.asarray(obs, dtype=np.float32), bool(terminated or truncated), dict(info)
    obs, reward, done, info = result
    return np.asarray(obs, dtype=np.float32), bool(done), dict(info)


def execute_probe(args: argparse.Namespace) -> dict[str, Any]:
    pairs_path = args.pairs_path or args.pair_jsonl
    if not pairs_path:
        raise ValueError("--pairs_path is required for execute mode")
    if not args.checkpoint_root or not args.env_name:
        raise ValueError("--checkpoint_root and --env_name are required for execute mode")
    out_jsonl = args.out_jsonl
    if not out_jsonl:
        raise ValueError("--out_jsonl is required for execute mode")

    rows = [r for r in iter_jsonl(pairs_path) if r.get("env_name") == args.env_name and int(r.get("seed", args.seed)) == int(args.seed)]
    probeable = [r for r in rows if r.get("probeable") and r.get("state_ref_s")]
    limit = args.num_pairs if args.max_pairs is None else args.max_pairs
    selected = probeable[: max(0, int(limit))]
    if not selected:
        result_rows = [
            {
                "record_type": "probe_failure",
                "env_name": args.env_name,
                "seed": args.seed,
                "failure_reason": "no exact-reset probeable pairs in input",
                "num_input_pairs": len(rows),
                "num_probeable_pairs": len(probeable),
            }
        ]
        write_jsonl(out_jsonl, result_rows)
        summary = summarize_probe_rows(result_rows)
        write_probe_summary(args.out_summary, summary)
        return summary

    env, agent, jax = setup_gas_agent(args)
    rng = jax.random.PRNGKey(args.seed)
    result_rows = []
    for i, row in enumerate(selected):
        pair_id = row.get("pair_id") or f"{row.get('path_idx')}::{row.get('pair_role')}::{row.get('path_position')}::{i}"
        try:
            state_ref = deserialize_state_ref(row["state_ref_s"])
            exact_reset = state_ref_is_exact(state_ref)
            if not exact_reset and not args.allow_approximate_reset:
                raise RuntimeError(f"pair is not exact-reset capable: {state_ref.reset_mode}")
            restore_env_from_state_ref(env, state_ref, allow_approximate=args.allow_approximate_reset)
            obs = current_observation(env)
            phi_g = np.asarray(row.get("phi_g") or row.get("g_ref"), dtype=np.float32)
            dists = []
            action_norms = []
            terminated = False
            truncated = False
            initial_phi = np.asarray(agent.get_phi(obs), dtype=np.float32)
            d_phi_start = float(np.linalg.norm(initial_phi - phi_g))
            min_d_phi = d_phi_start
            final_d_phi = d_phi_start
            policy_skill_norm = None
            for step in range(int(args.horizon)):
                phi_obs = np.asarray(agent.get_phi(obs), dtype=np.float32)
                delta = phi_g - phi_obs
                norm = float(np.linalg.norm(delta) + 1e-10)
                skill = delta / norm
                policy_skill_norm = float(np.linalg.norm(skill))
                rng, key = jax.random.split(rng)
                action = np.asarray(agent.sample_actions(observations=obs, goals=skill, temperature=0.0, seed=key))
                action = np.clip(action, -1, 1)
                action_norms.append(float(np.linalg.norm(action)))
                obs, done, info = step_env(env, action)
                phi_next = np.asarray(agent.get_phi(obs), dtype=np.float32)
                final_d_phi = float(np.linalg.norm(phi_next - phi_g))
                min_d_phi = min(min_d_phi, final_d_phi)
                dists.append(final_d_phi)
                terminated = bool(info.get("terminated", False)) or bool(done)
                truncated = bool(info.get("truncated", False))
                if done:
                    break
            threshold = float(row.get("eval_subgoal_threshold") or row.get("way_steps") or 1.0)
            delta_phi = d_phi_start - final_d_phi
            normalized_progress = delta_phi / (d_phi_start + 1e-10)
            result_rows.append(
                {
                    "record_type": "closed_loop_probe",
                    "pair_id": pair_id,
                    "env_name": args.env_name,
                    "seed": int(args.seed),
                    "pair_source": row.get("pair_role") or row.get("source"),
                    "path_position": row.get("path_position"),
                    "path_position_bucket": row.get("path_position_bucket"),
                    "final_phase": bool(row.get("final_phase")),
                    "recovery_candidate": bool(row.get("recovery_candidate", False)),
                    "q_train_support": row.get("q_train_support_score"),
                    "d_phi_start": d_phi_start,
                    "qg_d_phi": row.get("d_phi"),
                    "d_phi_end": final_d_phi,
                    "delta_phi": delta_phi,
                    "normalized_progress": normalized_progress,
                    "hit": bool(min_d_phi <= threshold),
                    "hit_threshold": threshold,
                    "min_d_phi_during_rollout": min_d_phi,
                    "final_d_phi": final_d_phi,
                    "action_norm_mean": float(np.mean(action_norms)) if action_norms else None,
                    "action_norm_max": float(np.max(action_norms)) if action_norms else None,
                    "policy_skill_norm": policy_skill_norm,
                    "rollout_length": len(action_norms),
                    "terminated": terminated,
                    "truncated": truncated,
                    "reset_mode": state_ref.reset_mode,
                    "exact_reset": exact_reset,
                    "failure_reason": None,
                    "phi_s": row.get("phi_s") or row.get("s_ref"),
                    "phi_g": row.get("phi_g") or row.get("g_ref"),
                }
            )
        except Exception as exc:
            result_rows.append(
                {
                    "record_type": "closed_loop_probe",
                    "pair_id": pair_id,
                    "env_name": args.env_name,
                    "seed": int(args.seed),
                    "pair_source": row.get("pair_role") or row.get("source"),
                    "path_position": row.get("path_position"),
                    "final_phase": bool(row.get("final_phase")),
                    "hit": False,
                    "exact_reset": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    write_jsonl(out_jsonl, result_rows)
    summary = summarize_probe_rows(result_rows)
    write_probe_summary(args.out_summary, summary)
    return summary


def summarize_probe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("record_type") == "closed_loop_probe" and not r.get("failure_reason")]
    return {
        "mode": "execute",
        "num_rows": len(rows),
        "num_valid": len(valid),
        "num_failed": len(rows) - len(valid),
        "hit_rate": float(np.mean([bool(r.get("hit")) for r in valid])) if valid else None,
        "mean_delta_phi": summarize_numeric(r.get("delta_phi") for r in valid)["mean"],
        "mean_normalized_progress": summarize_numeric(r.get("normalized_progress") for r in valid)["mean"],
        "negative_progress_rate": float(np.mean([(r.get("normalized_progress") or 0) < 0 for r in valid])) if valid else None,
        "action_norm_max_mean": summarize_numeric(r.get("action_norm_max") for r in valid)["mean"],
        "failure_reasons": summarize_failures(rows),
    }


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        reason = row.get("failure_reason")
        if reason:
            out[str(reason)] = out.get(str(reason), 0) + 1
    return out


def write_probe_summary(path: str | None, summary: dict[str, Any]) -> None:
    if not path:
        return
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP0 Closed-Loop Edge Probe Summary\n\n")
        for key, value in summary.items():
            fh.write(f"- `{key}`: {value}\n")


def write_trace_proxy_outputs(args: argparse.Namespace, result: dict[str, Any]) -> None:
    out_json = args.out_json or args.out_jsonl
    out_md = args.out_md or args.out_summary
    if out_json:
        write_json(out_json, result)
    if out_md:
        path = Path(out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write("# GP0/CLP0 Trace Proxy Probe\n\n")
            fh.write("## Correlations\n\n")
            for key, value in result.get("correlations", {}).items():
                fh.write(f"- `{key}`: {value if value is not None else 'NA'}\n")
            fh.write("\n")


def main() -> None:
    args = parse_args()
    mode = resolve_mode(args)
    if mode == "trace_proxy":
        result = trace_proxy_probe(args)
        write_trace_proxy_outputs(args, result)
        print({"mode": mode, "status": "ok", "num_joined_episodes": result.get("num_joined_episodes")})
    elif mode == "dry_run":
        print({"mode": mode, "pairs_path": args.pairs_path or args.pair_jsonl, "env_name": args.env_name})
    else:
        result = execute_probe(args)
        print({"mode": mode, "status": "ok", **{k: result.get(k) for k in ("num_rows", "num_valid", "hit_rate")}})


if __name__ == "__main__":
    main()

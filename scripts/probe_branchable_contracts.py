#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import iter_jsonl, write_jsonl
from probe_policy_edge_success import setup_gas_agent, current_observation, step_env, summarize_probe_rows, write_probe_summary

import sys

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.state_ref import deserialize_state_ref, restore_state_ref, is_exact_state_ref  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe frozen policy from exact branchable segment StateRefs.")
    parser.add_argument("--segments_path", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_segments", type=int, default=128)
    parser.add_argument("--horizons", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--target_modes", nargs="+", default=["original_target"])
    parser.add_argument("--gpu", default="")
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_summary", required=True)
    return parser.parse_args()


def target_for_mode(segment: dict[str, Any], mode: str) -> tuple[np.ndarray | None, str | None]:
    if mode == "original_target":
        return as_array(segment.get("target_phi")), None
    if mode == "recovery_candidate":
        if segment.get("target_source") == "recovery":
            return as_array(segment.get("target_phi")), None
        return None, "no recovery target recorded for segment"
    if mode == "final_goal":
        target = as_array(segment.get("final_goal_phi"))
        if target is not None:
            return target, None
        if segment.get("final_phase"):
            return as_array(segment.get("target_phi")), None
        return None, "final_goal phi was not recorded"
    if mode == "nearest_path_target":
        return path_target(segment, nearest=True)
    if mode == "farther_path_target":
        return path_target(segment, nearest=False)
    if mode == "qtrain_matched":
        return None, "qtrain matched target was not provided to branchable probe"
    return None, f"unsupported target_mode: {mode}"


def path_target(segment: dict[str, Any], nearest: bool) -> tuple[np.ndarray | None, str | None]:
    path = as_array(segment.get("path_phi"))
    start = as_array(segment.get("start_phi"))
    if path is None or path.ndim != 2:
        return None, "path_phi was not recorded in segment trace"
    if start is None:
        return None, "start_phi was not recorded in segment trace"
    position = segment.get("path_position")
    try:
        start_idx = max(0, int(position or 0))
    except (TypeError, ValueError):
        start_idx = 0
    suffix = path[start_idx:]
    if len(suffix) == 0:
        return None, "path suffix is empty"
    if nearest:
        idx = int(np.argmin(np.linalg.norm(suffix - start[None, :], axis=1)))
    else:
        idx = min(len(suffix) - 1, 2)
    return suffix[idx], None


def as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def main() -> None:
    args = parse_args()
    segments = [
        row for row in iter_jsonl(args.segments_path)
        if row.get("record_type") == "segment_contract"
        and row.get("env_name") == args.env_name
        and int(row.get("seed", args.seed)) == args.seed
    ]
    exact_segments = [s for s in segments if (s.get("start_state_ref") or {}).get("exact_reset")]
    selected = exact_segments[: args.num_segments]
    rows: list[dict[str, Any]] = []
    if selected:
        env, agent, jax = setup_gas_agent(args)
        rng = jax.random.PRNGKey(args.seed)
    else:
        env = agent = jax = rng = None
    for segment in selected:
        for mode in args.target_modes:
            target_phi, target_error = target_for_mode(segment, mode)
            for horizon in args.horizons:
                rows.append(probe_one(args, segment, mode, horizon, target_phi, target_error, env, agent, jax, rng))
    if not selected:
        rows.append({
            "record_type": "branchable_probe",
            "env_name": args.env_name,
            "seed": args.seed,
            "failure_reason": "no exact segment start_state_ref records",
            "skipped_exact_restore_false": len(segments),
        })
    write_jsonl(args.out_jsonl, rows)
    summary = summarize_probe_rows([normalize_for_summary(r) for r in rows])
    summary["num_segments_input"] = len(segments)
    summary["num_segments_exact"] = len(exact_segments)
    summary["target_modes"] = args.target_modes
    summary["horizons"] = args.horizons
    write_probe_summary(args.out_summary, summary)
    print({"out_jsonl": args.out_jsonl, "out_summary": args.out_summary, "rows": len(rows), "exact_segments": len(exact_segments)})


def probe_one(args, segment, mode, horizon, target_phi, target_error, env, agent, jax, rng):
    source_segment_id = segment.get("segment_id")
    probe_id = f"{source_segment_id}__{mode}__H{horizon}"
    base = {
        "record_type": "branchable_probe",
        "probe_id": probe_id,
        "source_segment_id": source_segment_id,
        "env_name": args.env_name,
        "seed": args.seed,
        "variant_source": segment.get("variant"),
        "target_mode": mode,
        "horizon": int(horizon),
        "path_position": segment.get("path_position"),
        "final_phase": bool(segment.get("final_phase")),
        "recovery_candidate": bool(mode == "recovery_candidate" or segment.get("target_source") == "recovery"),
        "q_train_support": segment.get("q_train_support"),
        "graph_d_phi": segment.get("d_phi_start"),
    }
    if target_error:
        return {**base, "hit": False, "exact_restore": False, "failure_reason": target_error}
    try:
        state_ref = deserialize_state_ref(segment["start_state_ref"])
        if not is_exact_state_ref(state_ref):
            raise RuntimeError(f"segment start is not exact: {state_ref.reset_mode}")
        restore_state_ref(env, state_ref)
        obs = current_observation(env)
        dists = []
        action_norms = []
        skill_norms = []
        terminated = False
        truncated = False
        time_to_hit = None
        phi0 = np.asarray(agent.get_phi(obs), dtype=np.float32)
        d_phi_start = float(np.linalg.norm(phi0 - target_phi))
        min_d_phi = d_phi_start
        final_d_phi = d_phi_start
        threshold = float(segment.get("hit_threshold") or 1.0)
        threshold = max(1e-6, threshold)
        stable_offset = int(hashlib.md5(probe_id.encode("utf-8")).hexdigest()[:8], 16) % 100000
        local_rng = jax.random.PRNGKey(args.seed + int(horizon) + stable_offset)
        for step in range(int(horizon)):
            phi_obs = np.asarray(agent.get_phi(obs), dtype=np.float32)
            delta = target_phi - phi_obs
            norm = float(np.linalg.norm(delta) + 1e-10)
            skill = delta / norm
            skill_norms.append(float(np.linalg.norm(skill)))
            local_rng, key = jax.random.split(local_rng)
            action = np.asarray(agent.sample_actions(observations=obs, goals=skill, temperature=0.0, seed=key))
            action = np.clip(action, -1, 1)
            action_norms.append(float(np.linalg.norm(action)))
            obs, done, info = step_env(env, action)
            phi_next = np.asarray(agent.get_phi(obs), dtype=np.float32)
            final_d_phi = float(np.linalg.norm(phi_next - target_phi))
            min_d_phi = min(min_d_phi, final_d_phi)
            dists.append(final_d_phi)
            if time_to_hit is None and min_d_phi <= threshold:
                time_to_hit = step + 1
            terminated = bool(info.get("terminated", False)) or bool(done)
            truncated = bool(info.get("truncated", False))
            if done:
                break
        delta_phi = d_phi_start - final_d_phi
        normalized_progress = delta_phi / (d_phi_start + 1e-10)
        return {
            **base,
            "exact_restore": True,
            "d_phi_start": d_phi_start,
            "d_phi_end": final_d_phi,
            "min_d_phi": min_d_phi,
            "delta_phi": delta_phi,
            "normalized_progress": normalized_progress,
            "hit": bool(time_to_hit is not None),
            "time_to_hit": time_to_hit,
            "negative_progress": bool(normalized_progress < 0),
            "action_norm_mean": float(np.mean(action_norms)) if action_norms else None,
            "action_norm_max": float(np.max(action_norms)) if action_norms else None,
            "skill_norm_mean": float(np.mean(skill_norms)) if skill_norms else None,
            "skill_norm_max": float(np.max(skill_norms)) if skill_norms else None,
            "terminated": terminated,
            "truncated": truncated,
            "failure_reason": None,
            "phi_start": segment.get("start_phi"),
            "phi_target": target_phi,
        }
    except Exception as exc:
        return {**base, "hit": False, "exact_restore": False, "failure_reason": f"{type(exc).__name__}: {exc}"}


def normalize_for_summary(row):
    return {
        "record_type": "closed_loop_probe",
        "hit": row.get("hit"),
        "failure_reason": row.get("failure_reason"),
        "delta_phi": row.get("delta_phi"),
        "normalized_progress": row.get("normalized_progress"),
        "action_norm_max": row.get("action_norm_max"),
    }


if __name__ == "__main__":
    main()

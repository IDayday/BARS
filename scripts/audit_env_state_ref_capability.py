#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import numpy as np

from cage_gp0_common import write_json

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.state_ref import capture_state_ref, compare_state_ref_restore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact branchable StateRef support for OGBench/GAS envs.")
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--restore_tolerance", type=float, default=1e-5)
    return parser.parse_args()


def current_obs(env: Any) -> np.ndarray:
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "get_ob"):
        return np.asarray(unwrapped.get_ob(), dtype=np.float32)
    if hasattr(unwrapped, "state_vector"):
        return np.asarray(unwrapped.state_vector(), dtype=np.float32)
    raise RuntimeError("env does not expose get_ob/state_vector")


def audit_env(env_name: str, tolerance: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "env_name": env_name,
        "can_construct": False,
        "reset_seed_works": False,
        "observation_shape": None,
        "action_shape": None,
        "qpos_qvel_available": False,
        "has_unwrapped_set_state": False,
        "has_sim_set_state": False,
        "has_mujoco_data_qpos_qvel": False,
        "dataset_or_env_exposes_state_ids": False,
        "dataset_keys": [],
        "restore_test_ran": False,
        "restore_test_passed": False,
        "max_abs_obs_error": None,
        "max_abs_phi_error": None,
        "recommended_state_ref_mode": "unsupported",
        "notes": [],
    }
    try:
        import ogbench  # type: ignore
    except Exception as exc:
        row["error"] = f"ogbench import failed: {type(exc).__name__}: {exc}"
        return row
    try:
        env, train_dataset, _ = ogbench.make_env_and_datasets(env_name, compact_dataset=False)
        row["can_construct"] = True
        row["dataset_keys"] = sorted(train_dataset.keys())
    except Exception as exc:
        row["error"] = f"env construction failed: {type(exc).__name__}: {exc}"
        return row
    try:
        obs, _ = env.reset(seed=0)
        row["reset_seed_works"] = True
        row["observation_shape"] = list(np.asarray(obs).shape)
        row["action_shape"] = list(getattr(env.action_space, "shape", ()) or ())
    except Exception as exc:
        row["error"] = f"reset failed: {type(exc).__name__}: {exc}"
        return row

    unwrapped = getattr(env, "unwrapped", env)
    data = getattr(unwrapped, "data", None)
    row["has_unwrapped_set_state"] = bool(hasattr(unwrapped, "set_state"))
    row["has_sim_set_state"] = bool(hasattr(getattr(unwrapped, "sim", None), "set_state"))
    row["has_mujoco_data_qpos_qvel"] = bool(data is not None and hasattr(data, "qpos") and hasattr(data, "qvel"))
    row["qpos_qvel_available"] = bool(row["has_mujoco_data_qpos_qvel"])
    row["dataset_or_env_exposes_state_ids"] = any(k in train_dataset for k in ("state_ids", "states", "sim_states", "qpos", "qvel"))

    if not (row["has_unwrapped_set_state"] and row["has_mujoco_data_qpos_qvel"]):
        row["recommended_state_ref_mode"] = "unsupported"
        row["notes"].append("Missing set_state or qpos/qvel access.")
        return row

    try:
        rng = np.random.default_rng(0)
        env.reset(seed=0)
        for _ in range(3):
            env.step(rng.uniform(low=-0.25, high=0.25, size=env.action_space.shape).astype(np.float32))
        captured_obs = current_obs(env)
        state_ref = capture_state_ref(
            env,
            obs=captured_obs,
            phi=None,
            metadata={"env_name": env_name, "source": "rollout_segment", "source_variant": "audit"},
        )
        for _ in range(3):
            env.step(rng.uniform(low=-0.25, high=0.25, size=env.action_space.shape).astype(np.float32))
        comparison = compare_state_ref_restore(env, state_ref)
        row["restore_test_ran"] = True
        row["max_abs_obs_error"] = comparison.get("max_abs_obs_error")
        row["max_abs_phi_error"] = comparison.get("max_abs_phi_error")
        row["restore_test_passed"] = bool(
            comparison.get("restored")
            and row["max_abs_obs_error"] is not None
            and float(row["max_abs_obs_error"]) <= tolerance
        )
        row["recommended_state_ref_mode"] = "exact_mujoco_state" if row["restore_test_passed"] else "unsupported"
        if not row["restore_test_passed"]:
            row["notes"].append(f"Restore test failed: {comparison}")
    except Exception as exc:
        row["restore_test_ran"] = True
        row["recommended_state_ref_mode"] = "unsupported"
        row["error"] = f"restore test failed: {type(exc).__name__}: {exc}"
    return row


def write_md(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "env_name",
        "can_construct",
        "reset_seed_works",
        "observation_shape",
        "action_shape",
        "qpos_qvel_available",
        "has_unwrapped_set_state",
        "has_sim_set_state",
        "has_mujoco_data_qpos_qvel",
        "restore_test_passed",
        "max_abs_obs_error",
        "dataset_or_env_exposes_state_ids",
        "recommended_state_ref_mode",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP1 StateRef Capability Audit\n\n")
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for row in rows:
            vals = []
            for col in cols:
                val = row.get(col)
                vals.append("NA" if val is None else (f"{val:.6g}" if isinstance(val, float) else str(val)))
            fh.write("| " + " | ".join(vals) + " |\n")
        fh.write("\n## Per-Env Notes\n\n")
        for row in rows:
            fh.write(f"### {row['env_name']}\n\n")
            if row.get("error"):
                fh.write(f"- Error: {row['error']}\n")
            for note in row.get("notes", []):
                fh.write(f"- {note}\n")
            if not row.get("error") and not row.get("notes"):
                fh.write("- Exact restore test passed.\n")
            fh.write("\n")


def main() -> None:
    args = parse_args()
    rows = [audit_env(env_name, args.restore_tolerance) for env_name in args.envs]
    write_json(args.out_json, {"envs": rows})
    write_md(args.out_md, rows)
    print({"out_json": args.out_json, "out_md": args.out_md, "envs": len(rows)})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether GAS/OGBench envs support exact state reset for CLP0 probes.")
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    return parser.parse_args()


def audit_env(env_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "env_name": env_name,
        "can_construct": False,
        "reset_seed_works": False,
        "observation_shape": None,
        "action_shape": None,
        "has_qpos_qvel_attrs": False,
        "has_unwrapped_set_state": False,
        "has_sim_set_state": False,
        "has_mujoco_data_qpos_qvel": False,
        "qpos_shape": None,
        "qvel_shape": None,
        "dataset_keys": [],
        "dataset_has_qpos": False,
        "dataset_has_qvel": False,
        "dataset_has_state": False,
        "dataset_contains_raw_qpos_qvel_or_state": False,
        "observation_equals_qpos_qvel": False,
        "observation_sufficient_to_reconstruct_state": False,
        "recommended_reset_mode": "unsupported",
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
    except Exception as exc:
        row["error"] = f"env construction failed: {type(exc).__name__}: {exc}"
        return row

    try:
        obs, info = env.reset(seed=0)
        row["reset_seed_works"] = True
        row["observation_shape"] = list(np.asarray(obs).shape)
        row["action_shape"] = list(getattr(env.action_space, "shape", ()) or ())
    except Exception as exc:
        row["error"] = f"reset failed: {type(exc).__name__}: {exc}"
        return row

    unwrapped = getattr(env, "unwrapped", env)
    row["has_qpos_qvel_attrs"] = bool(hasattr(unwrapped, "qpos") and hasattr(unwrapped, "qvel"))
    row["has_unwrapped_set_state"] = bool(hasattr(unwrapped, "set_state"))
    row["has_sim_set_state"] = bool(hasattr(getattr(unwrapped, "sim", None), "set_state"))
    data = getattr(unwrapped, "data", None)
    if data is not None and hasattr(data, "qpos") and hasattr(data, "qvel"):
        qpos = np.asarray(data.qpos)
        qvel = np.asarray(data.qvel)
        row["has_mujoco_data_qpos_qvel"] = True
        row["qpos_shape"] = list(qpos.shape)
        row["qvel_shape"] = list(qvel.shape)
        concat = np.concatenate([qpos.ravel(), qvel.ravel()])
        obs_arr = np.asarray(obs).ravel()
        row["observation_equals_qpos_qvel"] = bool(obs_arr.shape == concat.shape and np.allclose(obs_arr, concat))
    keys = sorted(train_dataset.keys())
    row["dataset_keys"] = keys
    row["dataset_has_qpos"] = "qpos" in train_dataset
    row["dataset_has_qvel"] = "qvel" in train_dataset
    row["dataset_has_state"] = any(k in train_dataset for k in ("states", "state", "sim_states", "infos/qpos", "infos/qvel"))
    row["dataset_contains_raw_qpos_qvel_or_state"] = bool(row["dataset_has_qpos"] and row["dataset_has_qvel"] or row["dataset_has_state"])

    exact_runtime = bool(row["has_unwrapped_set_state"] and row["has_mujoco_data_qpos_qvel"])
    if row["dataset_contains_raw_qpos_qvel_or_state"]:
        row["recommended_reset_mode"] = "exact_mujoco_state"
        row["observation_sufficient_to_reconstruct_state"] = True
    elif row["observation_equals_qpos_qvel"] and exact_runtime:
        row["recommended_reset_mode"] = "dataset_state_ref"
        row["observation_sufficient_to_reconstruct_state"] = True
        row["notes"].append("Dataset observation is exactly qpos||qvel for this env; dataset observations can be converted to exact StateRef.")
    elif exact_runtime:
        row["recommended_reset_mode"] = "observation_only_not_exact"
        row["notes"].append("Runtime env can capture exact qpos/qvel, but saved dataset observations do not reconstruct qpos/qvel.")
    else:
        row["recommended_reset_mode"] = "unsupported"
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
        "has_unwrapped_set_state",
        "has_mujoco_data_qpos_qvel",
        "dataset_contains_raw_qpos_qvel_or_state",
        "observation_equals_qpos_qvel",
        "recommended_reset_mode",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP0 Reset Capability Audit\n\n")
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row.get(c)) for c in cols) + " |\n")
        fh.write("\n## Notes\n\n")
        for row in rows:
            fh.write(f"### {row['env_name']}\n\n")
            for note in row.get("notes", []):
                fh.write(f"- {note}\n")
            if row.get("error"):
                fh.write(f"- Error: {row['error']}\n")
            if not row.get("notes") and not row.get("error"):
                fh.write("- No additional notes.\n")
            fh.write("\n")


def main() -> None:
    args = parse_args()
    rows = [audit_env(env_name) for env_name in args.envs]
    write_json(args.out_json, {"envs": rows})
    write_md(args.out_md, rows)
    print({"out_json": args.out_json, "out_md": args.out_md, "envs": len(rows)})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Probe reset-to-state support for Phase 3 edge rollout validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data import load_ogbench_dataset  # noqa: E402
from phase3.reset_utils import (  # noqa: E402
    DEFAULT_RECONSTRUCTION_TOLERANCE,
    env_unavailable_probe_result,
    missing_reset_env_packages,
    probe_reset_capability_many,
)


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
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


def _value_at(dataset: dict[str, Any], key: str, idx: int) -> Any | None:
    if key not in dataset:
        return None
    value = dataset[key]
    if not isinstance(value, np.ndarray) or value.ndim == 0 or value.shape[0] <= idx:
        return None
    return value[idx]


def _probe_state(dataset: dict[str, Any], idx: int) -> dict[str, Any]:
    state: dict[str, Any] = {"observation": dataset["observations"][idx]}
    for key in ("qpos", "qvel", "state", "states", "sim_state", "sim_states", "infos/qpos", "infos/qvel"):
        value = _value_at(dataset, key, idx)
        if value is not None:
            state[key] = value
    return state


def _sample_indices(n: int, num_probe_states: int, seed: int) -> np.ndarray:
    if n <= 0 or num_probe_states <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    take = min(int(num_probe_states), int(n))
    return np.sort(rng.choice(n, size=take, replace=False)).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_dir", default="/mnt/project/offlinerl_datasets/ogbench")
    parser.add_argument("--output_dir", default="results/phase3")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--num_probe_states", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reconstruction_tolerance", type=float, default=DEFAULT_RECONSTRUCTION_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split=args.split,
        max_transitions=args.max_transitions,
    )
    observations = np.asarray(dataset["observations"])
    indices = _sample_indices(observations.shape[0], args.num_probe_states, args.seed)
    states = [_probe_state(dataset, int(idx)) for idx in indices]
    out_dir = Path(args.output_dir) / _dataset_key(args.dataset_name) / "reset_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = _make_env(args.dataset_name, args.dataset_dir)
    except Exception as exc:
        summary = {
            "dataset_name": args.dataset_name,
            **env_unavailable_probe_result(
                f"env_construction_failed: {type(exc).__name__}: {exc}",
                missing_packages=missing_reset_env_packages(),
                num_probe_states=int(indices.size),
            ),
        }
        examples = [
            {
                "probe_index": int(i),
                "dataset_index": int(idx),
                "env_available": False,
                "reset_probe_status": summary["reset_probe_status"],
                "reset_supported": None,
                "reset_method": None,
                "obs_reconstruction_error": np.nan,
                "failure_reason": summary["failure_reason"],
                "attempted_methods": "",
                "missing_packages": " | ".join(summary["missing_packages"]),
            }
            for i, idx in enumerate(indices)
        ]
        _write_json(out_dir / "reset_probe_summary.json", summary)
        import pandas as pd

        pd.DataFrame(examples).to_csv(out_dir / "reset_probe_examples.csv", index=False)
        print(f"[phase3B] reset_probe_status={summary['reset_probe_status']} output_dir={out_dir}")
        print(json.dumps(_json_safe(summary), sort_keys=True))
        return

    summary, examples_df = probe_reset_capability_many(
        env,
        states,
        reconstruction_tolerance=args.reconstruction_tolerance,
    )
    summary = {
        "dataset_name": args.dataset_name,
        **summary,
    }
    examples_df = examples_df.copy()
    examples_df.insert(1, "dataset_index", indices[: examples_df.shape[0]])
    _write_json(out_dir / "reset_probe_summary.json", summary)
    examples_df.to_csv(out_dir / "reset_probe_examples.csv", index=False)
    print(f"[phase3B] reset_probe_status={summary['reset_probe_status']} output_dir={out_dir}")
    print(json.dumps(_json_safe(summary), sort_keys=True))


if __name__ == "__main__":
    main()

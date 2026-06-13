#!/usr/bin/env python
"""Phase 3 environment preflight before reset probing and rollouts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3.reset_utils import (  # noqa: E402
    DEFAULT_RECONSTRUCTION_TOLERANCE,
    RESET_STATUS_ENV_UNAVAILABLE,
    RESET_STATUS_UNCERTAIN,
    probe_reset_capability_many,
)


PACKAGE_SPECS = {
    "gymnasium": {"optional": False},
    "gym": {"optional": False},
    "ogbench": {"optional": False},
    "mujoco": {"optional": False},
    "d4rl": {"optional": True},
}


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


def _package_version(module: Any) -> str | None:
    version = getattr(module, "__version__", None)
    return None if version is None else str(version)


def check_package_imports() -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any | None]]:
    package_imports: dict[str, dict[str, Any]] = {}
    modules: dict[str, Any | None] = {}
    missing_packages: list[str] = []
    for name, spec in PACKAGE_SPECS.items():
        optional = bool(spec["optional"])
        try:
            module = importlib.import_module(name)
            modules[name] = module
            package_imports[name] = {
                "imported": True,
                "optional": optional,
                "version": _package_version(module),
                "error": None,
            }
        except Exception as exc:  # noqa: PERF203 - each import is an independent preflight check.
            modules[name] = None
            missing_dependency = getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else name
            package_imports[name] = {
                "imported": False,
                "optional": optional,
                "version": None,
                "error": f"{type(exc).__name__}: {exc}",
                "missing_dependency": missing_dependency,
            }
            if not optional and missing_dependency:
                missing_packages.append(str(missing_dependency))
    return package_imports, sorted(set(missing_packages)), modules


def _dataset_keys(dataset: Any) -> list[str]:
    if dataset is None:
        return []
    if hasattr(dataset, "keys"):
        try:
            return sorted(str(k) for k in dataset.keys())
        except Exception:
            pass
    if hasattr(dataset, "items"):
        try:
            return sorted(str(k) for k, _ in dataset.items())
        except Exception:
            pass
    return sorted(k for k in vars(dataset).keys() if not k.startswith("_"))


def _get_item(dataset: Any, key: str) -> Any:
    try:
        return dataset[key]
    except (KeyError, TypeError):
        return getattr(dataset, key)


def _value_at(dataset: Any, key: str, idx: int) -> Any | None:
    try:
        value = _get_item(dataset, key)
    except (AttributeError, KeyError, TypeError):
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim == 0 or arr.shape[0] <= idx:
        return None
    return arr[idx]


def _probe_state(dataset: Any, idx: int) -> dict[str, Any]:
    state: dict[str, Any] = {"observation": _get_item(dataset, "observations")[idx]}
    for key in ("qpos", "qvel", "state", "states", "sim_state", "sim_states", "infos/qpos", "infos/qvel"):
        value = _value_at(dataset, key, idx)
        if value is not None:
            state[key] = value
    return state


def _sample_indices(dataset: Any, num_probe_states: int, seed: int) -> np.ndarray:
    try:
        observations = np.asarray(_get_item(dataset, "observations"))
    except Exception:
        return np.empty(0, dtype=np.int64)
    if observations.ndim == 0 or observations.shape[0] <= 0 or num_probe_states <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    take = min(int(num_probe_states), int(observations.shape[0]))
    return np.sort(rng.choice(observations.shape[0], size=take, replace=False)).astype(np.int64)


def _recommended_install_hint(missing_packages: list[str], env_error_trace: str | None) -> str:
    hints: list[str] = []
    missing = set(missing_packages)
    if "ogbench" in missing:
        hints.append("Install or expose the ogbench Python package in this environment.")
    if "gymnasium" in missing and "gym" in missing:
        hints.append("Install gymnasium or gym plus the backend extras required by the target OGBench env.")
    if "mujoco" in missing:
        hints.append("Install mujoco if the target environment uses MuJoCo physics.")
    if env_error_trace:
        hints.append(
            "Verify that ogbench.make_env_and_datasets(dataset_name, dataset_dir=...) works in the same Python env."
        )
    hints.append("d4rl is checked for compatibility only; a missing d4rl import is not fatal for this preflight.")
    return " ".join(hints)


def _remove_stale_reset_outputs(out_dir: Path) -> None:
    for name in ("reset_probe_summary.json", "reset_probe_examples.csv"):
        path = out_dir / name
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_dir", default="/mnt/project/offlinerl_datasets/ogbench")
    parser.add_argument("--output_dir", default="results/phase3/env_preflight")
    parser.add_argument("--num_probe_states", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reconstruction_tolerance", type=float, default=DEFAULT_RECONSTRUCTION_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / _dataset_key(args.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    package_imports, missing_packages, modules = check_package_imports()
    ogbench_imported = bool(package_imports["ogbench"]["imported"])
    env = None
    train_dataset = None
    val_dataset = None
    env_error_trace = None
    ogbench_env_constructed = False
    dataset_loaded = False
    train_dataset_keys: list[str] = []

    if ogbench_imported:
        try:
            env, train_dataset, val_dataset = modules["ogbench"].make_env_and_datasets(
                args.dataset_name,
                dataset_dir=args.dataset_dir,
                compact_dataset=False,
            )
            del val_dataset
            ogbench_env_constructed = True
            dataset_loaded = train_dataset is not None
            train_dataset_keys = _dataset_keys(train_dataset)
        except Exception:
            env_error_trace = traceback.format_exc()
    else:
        env_error_trace = package_imports["ogbench"]["error"]

    recommended_install_hint = _recommended_install_hint(missing_packages, env_error_trace)

    summary: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "status": "env_available" if ogbench_env_constructed else RESET_STATUS_ENV_UNAVAILABLE,
        "package_imports": package_imports,
        "missing_packages": missing_packages,
        "ogbench_imported": ogbench_imported,
        "ogbench_env_constructed": ogbench_env_constructed,
        "dataset_loaded": dataset_loaded,
        "train_dataset_keys": train_dataset_keys,
        "env_error_trace": env_error_trace,
        "recommended_install_hint": recommended_install_hint,
    }

    if not ogbench_env_constructed:
        _remove_stale_reset_outputs(out_dir)
        _write_json(out_dir / "preflight_summary.json", summary)
        print(f"[phase3-preflight] status={summary['status']} output_dir={out_dir}")
        print(json.dumps(_json_safe(summary), sort_keys=True))
        return

    indices = _sample_indices(train_dataset, args.num_probe_states, args.seed)
    if indices.size == 0:
        reset_summary = {
            "env_available": True,
            "env_construction_error": None,
            "reset_probe_status": RESET_STATUS_UNCERTAIN,
            "reset_supported": None,
            "reset_method": None,
            "obs_reconstruction_error_mean": np.nan,
            "obs_reconstruction_error_max": np.nan,
            "num_probe_states": 0,
            "missing_packages": [],
            "failure_reason": "train_dataset_missing_observations",
        }
        import pandas as pd

        reset_examples = pd.DataFrame()
    else:
        states = [_probe_state(train_dataset, int(idx)) for idx in indices]
        reset_summary, reset_examples = probe_reset_capability_many(
            env,
            states,
            reconstruction_tolerance=args.reconstruction_tolerance,
        )
        reset_examples = reset_examples.copy()
        reset_examples.insert(1, "dataset_index", indices[: reset_examples.shape[0]])

    reset_summary = {
        "dataset_name": args.dataset_name,
        **reset_summary,
    }
    _write_json(out_dir / "reset_probe_summary.json", reset_summary)
    reset_examples.to_csv(out_dir / "reset_probe_examples.csv", index=False)
    summary.update(
        {
            "reset_probe_status": reset_summary["reset_probe_status"],
            "reset_supported": reset_summary["reset_supported"],
            "reset_method": reset_summary["reset_method"],
            "reset_probe_summary_path": str(out_dir / "reset_probe_summary.json"),
        }
    )
    _write_json(out_dir / "preflight_summary.json", summary)
    print(
        "[phase3-preflight] "
        f"status={summary['status']} reset_probe_status={summary['reset_probe_status']} output_dir={out_dir}"
    )
    print(json.dumps(_json_safe(summary), sort_keys=True))


if __name__ == "__main__":
    main()

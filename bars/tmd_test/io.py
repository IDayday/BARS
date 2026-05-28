from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_DATASET_ROOT = Path(os.environ.get("BARS_TMD_TEST_DATASET_ROOT", "/mnt/project/offlinerl_datasets/ogbench"))


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def locate_dataset_npz(env_name: str, dataset_root: str | os.PathLike[str] | None = None) -> Path:
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    candidates = [root / f"{env_name}.npz", root / "ogbench" / f"{env_name}.npz", Path("_data") / "ogbench" / f"{env_name}.npz"]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(f"Could not find {env_name}.npz. Checked: " + ", ".join(str(p) for p in candidates))


def locate_val_npz(env_name: str, dataset_root: str | os.PathLike[str] | None = None) -> Path | None:
    train = locate_dataset_npz(env_name, dataset_root)
    val = train.with_name(train.stem + "-val.npz")
    return val if val.exists() and val.stat().st_size > 0 else None


def load_raw_ogbench_npz(env_name: str, dataset_root: str | os.PathLike[str] | None = None, max_observations: int | None = None) -> dict[str, Any]:
    path = locate_dataset_npz(env_name, dataset_root)
    with np.load(path) as z:
        observations = z["observations"].astype(np.float32, copy=False)
        actions = z["actions"].astype(np.float32, copy=False)
        terminals = z["terminals"].astype(bool, copy=False)
    if max_observations is not None and max_observations > 0 and max_observations < len(observations):
        observations = observations[:max_observations]
        actions = actions[:max_observations]
        terminals = terminals[:max_observations].copy()
        terminals[-1] = True
    return {"env_name": env_name, "dataset_path": str(path), "observations": observations, "actions": actions, "terminals": terminals}


def split_trajectories(terminals: np.ndarray) -> list[tuple[int, int]]:
    terminals = np.asarray(terminals).astype(bool)
    out: list[tuple[int, int]] = []
    start = 0
    for i, done in enumerate(terminals):
        if done:
            if i + 1 - start >= 2:
                out.append((start, i + 1))
            start = i + 1
    if len(terminals) - start >= 2:
        out.append((start, len(terminals)))
    return out


def terminal_goal_observations(env_name: str, dataset_root: str | os.PathLike[str] | None = None, max_goals: int = 5) -> np.ndarray:
    val = locate_val_npz(env_name, dataset_root)
    path = val if val is not None else locate_dataset_npz(env_name, dataset_root)
    with np.load(path) as z:
        obs = z["observations"].astype(np.float32, copy=False)
        terminals = z["terminals"].astype(bool, copy=False)
    idx = np.flatnonzero(terminals)
    if len(idx) == 0:
        idx = np.linspace(0, len(obs) - 1, num=min(max_goals, len(obs)), dtype=np.int64)
    return obs[idx[:max_goals]].copy()

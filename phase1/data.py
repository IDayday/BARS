from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = ("observations", "actions", "next_observations", "terminals")


def _dataset_file_name(dataset_name: str, split: str) -> str:
    suffix = "" if split == "train" else "-val"
    return f"{dataset_name}{suffix}.npz"


def _dataset_keys(dataset: Any) -> list[str]:
    if hasattr(dataset, "keys"):
        return list(dataset.keys())
    if hasattr(dataset, "items"):
        return [k for k, _ in dataset.items()]
    return [k for k in vars(dataset).keys() if not k.startswith("_")]


def _get_item(dataset: Any, key: str) -> Any:
    try:
        return dataset[key]
    except (TypeError, KeyError):
        return getattr(dataset, key)


def _as_array_if_possible(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.dtype == object and arr.ndim == 0:
        return value
    return arr


def _to_plain_dict(dataset: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _dataset_keys(dataset):
        out[key] = _as_array_if_possible(_get_item(dataset, key))
    return out


def _truncate_arrays(dataset: dict[str, Any], max_transitions: int | None) -> dict[str, Any]:
    if max_transitions is None:
        return dataset
    truncated: dict[str, Any] = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            truncated[key] = value[:max_transitions]
        else:
            truncated[key] = value
    return truncated


def _with_next_observations(dataset: dict[str, Any]) -> dict[str, Any]:
    if "next_observations" in dataset:
        return dataset
    observations = np.asarray(dataset["observations"])
    terminals = np.asarray(dataset.get("terminals", np.zeros(observations.shape[0], dtype=bool))).reshape(-1).astype(bool)
    next_observations = np.empty_like(observations)
    if observations.shape[0] == 0:
        dataset["next_observations"] = next_observations
        return dataset
    next_observations[:-1] = observations[1:]
    next_observations[-1] = observations[-1]
    terminal_idx = np.flatnonzero(terminals[: observations.shape[0]])
    next_observations[terminal_idx] = observations[terminal_idx]
    dataset["next_observations"] = next_observations
    return dataset


def _load_local_npz_dataset(
    dataset_name: str,
    dataset_dir: str | None,
    split: str,
    max_transitions: int | None,
) -> dict[str, Any]:
    root = Path(dataset_dir or "/mnt/project/offlinerl_datasets/ogbench").expanduser()
    path = root / _dataset_file_name(dataset_name, split)
    if not path.exists():
        raise FileNotFoundError(f"Local OGBench npz not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        dataset = {key: np.asarray(data[key]) for key in data.files}
    dataset = _with_next_observations(_truncate_arrays(dataset, max_transitions))
    missing = [key for key in REQUIRED_KEYS if key not in dataset]
    if missing:
        raise KeyError(f"Local OGBench npz is missing required keys: {missing}")
    print(f"[phase1] loaded local npz dataset={dataset_name!r} split={split!r} path={path}")
    return dataset


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, np.ndarray) else None


def _dim_from_shape(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    if len(shape) == 1:
        return 1
    return int(np.prod(shape[1:]))


def dataset_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    observations = np.asarray(dataset["observations"])
    actions = np.asarray(dataset["actions"])
    terminals = np.asarray(dataset["terminals"]).reshape(-1)
    return {
        "num_transitions": int(observations.shape[0]),
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "observation_dim": _dim_from_shape(observations.shape),
        "action_dim": _dim_from_shape(actions.shape),
        "num_terminal_flags": int(np.count_nonzero(terminals)),
        "keys": sorted(dataset.keys()),
        "has_next_observations": "next_observations" in dataset,
        "has_masks": "masks" in dataset,
        "has_valids": "valids" in dataset,
    }


def load_ogbench_dataset(
    dataset_name: str,
    dataset_dir: str | None,
    split: str = "train",
    max_transitions: int | None = None,
) -> dict[str, Any]:
    """Load an OGBench dataset, falling back to local offline npz arrays."""

    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    try:
        import ogbench

        _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_dir=dataset_dir,
            compact_dataset=False,
        )
        selected = train_dataset if split == "train" else val_dataset
        dataset = _with_next_observations(_truncate_arrays(_to_plain_dict(selected), max_transitions))
    except Exception as exc:
        print(
            "[phase1] official ogbench load unavailable; using local npz fallback "
            f"for dataset={dataset_name!r} split={split!r}: {type(exc).__name__}: {exc}"
        )
        dataset = _load_local_npz_dataset(dataset_name, dataset_dir, split, max_transitions)

    missing = [key for key in REQUIRED_KEYS if key not in dataset]
    if missing:
        raise KeyError(f"OGBench dataset is missing required keys: {missing}")

    summary = dataset_summary(dataset)
    print(f"[phase1] loaded dataset={dataset_name!r} split={split!r}")
    print(f"[phase1] keys={summary['keys']}")
    for key in summary["keys"]:
        value = dataset[key]
        if isinstance(value, np.ndarray):
            print(f"[phase1] {key}: shape={value.shape} dtype={value.dtype}")
    print(
        "[phase1] observation_dim="
        f"{summary['observation_dim']} action_dim={summary['action_dim']}"
    )
    return dataset


def save_dataset_summary(dataset: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(dataset)
    with (output_path / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary

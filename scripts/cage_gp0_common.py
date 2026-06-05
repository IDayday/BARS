from __future__ import annotations

import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_GP0_ENVS = [
    "antmaze-giant-navigate-v0",
    "antmaze-giant-stitch-v0",
    "humanoidmaze-large-navigate-v0",
]


def env_seed_root(checkpoint_root: str | Path, env_name: str, seed: int) -> Path:
    return Path(checkpoint_root) / env_name / f"seed{int(seed)}"


def keygraph_path(checkpoint_root: str | Path, env_name: str, seed: int) -> Path:
    return env_seed_root(checkpoint_root, env_name, seed) / "graph" / "keygraph.pkl"


def embeddings_path(checkpoint_root: str | Path, env_name: str, seed: int) -> Path:
    return env_seed_root(checkpoint_root, env_name, seed) / "features" / "dataset_embeddings.npy"


def policy_path(checkpoint_root: str | Path, env_name: str, seed: int) -> Path:
    return env_seed_root(checkpoint_root, env_name, seed) / "policy" / "params_1000000.pkl"


def manifest_path(checkpoint_root: str | Path, env_name: str, seed: int) -> Path:
    return env_seed_root(checkpoint_root, env_name, seed) / "manifest.json"


def load_keygraph(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        data = data.__dict__
    return data


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=2, sort_keys=True)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
            count += 1
    return count


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def finite(values: Iterable[Any]) -> np.ndarray:
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    return np.asarray(vals, dtype=np.float64)


def summarize_numeric(values: Iterable[Any]) -> dict[str, Any]:
    arr = finite(values)
    if len(arr) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p50": None, "p90": None, "max": None}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
    }


def group_rows(rows: Iterable[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    return groups


def select_task_ids(keygraph: dict[str, Any], requested: list[int] | None = None) -> list[int]:
    ids = sorted(int(k) for k in keygraph.get("task_paths_dict", {}).keys())
    if requested:
        keep = {int(x) for x in requested}
        ids = [x for x in ids if x in keep]
    return ids


def shortest_path_indices(keygraph: dict[str, Any], task_id: int, source: np.ndarray, force_closest: bool = True) -> tuple[list[int] | None, float | None]:
    nodes = np.asarray(keygraph["nodes"])
    shortest_paths = keygraph["task_paths_dict"][task_id]
    shortest_paths_dist = keygraph["task_paths_dist_dict"][task_id]
    sp_keys = list(shortest_paths.keys())
    if not sp_keys:
        return None, None
    start_distances = np.linalg.norm(nodes[sp_keys] - source, axis=1)
    valid_indices = np.where(start_distances <= float(keygraph["way_steps"]))[0]
    if len(valid_indices) == 0:
        if not force_closest:
            return None, None
        valid_indices = [int(np.argmin(start_distances))]
    best_total = float("inf")
    best_path = None
    for idx in valid_indices:
        key = sp_keys[int(idx)]
        total = float(start_distances[int(idx)] + shortest_paths_dist[key])
        if total < best_total:
            best_total = total
            best_path = [int(x) for x in shortest_paths[key]]
    return best_path, best_total


def gas_initial_target_index(path_nodes: list[int], nodes: np.ndarray, source: np.ndarray, subgoal_threshold: float) -> int:
    if not path_nodes:
        return -1
    path = nodes[path_nodes]
    distances = np.linalg.norm(path - source, axis=1)
    valid = np.where(distances <= subgoal_threshold)[0]
    return int(valid[-1]) if len(valid) else 0


def path_position_bucket(position: int | None, path_length: int | None, final_phase: bool = False) -> str:
    if final_phase:
        return "final"
    if position is None or path_length in (None, 0):
        return "unknown"
    if position == 0:
        return "initial"
    frac = position / max(path_length - 1, 1)
    if frac < 0.33:
        return "early"
    if frac < 0.67:
        return "mid"
    return "late"


def pair_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    s_list = []
    g_list = []
    idxs = []
    for i, row in enumerate(rows):
        if "s_ref" not in row or "g_ref" not in row:
            continue
        s_list.append(np.asarray(row["s_ref"], dtype=np.float32))
        g_list.append(np.asarray(row["g_ref"], dtype=np.float32))
        idxs.append(i)
    if not s_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []
    return np.stack(s_list), np.stack(g_list), idxs


def nearest_pair_distance(
    query_s: np.ndarray,
    query_g: np.ndarray,
    ref_s: np.ndarray,
    ref_g: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    if len(query_s) == 0 or len(ref_s) == 0:
        return np.full((len(query_s),), np.inf, dtype=np.float32)
    out = np.empty((len(query_s),), dtype=np.float32)
    ref_s = ref_s.astype(np.float32, copy=False)
    ref_g = ref_g.astype(np.float32, copy=False)
    for start in range(0, len(query_s), batch_size):
        end = min(start + batch_size, len(query_s))
        qs = query_s[start:end].astype(np.float32, copy=False)
        qg = query_g[start:end].astype(np.float32, copy=False)
        best = np.full((end - start,), np.inf, dtype=np.float32)
        ref_batch = max(1, min(8192, len(ref_s)))
        for r0 in range(0, len(ref_s), ref_batch):
            r1 = min(r0 + ref_batch, len(ref_s))
            ds = np.sum((qs[:, None, :] - ref_s[None, r0:r1, :]) ** 2, axis=-1)
            dg = np.sum((qg[:, None, :] - ref_g[None, r0:r1, :]) ** 2, axis=-1)
            best = np.minimum(best, np.sqrt(np.min(ds + dg, axis=1)))
        out[start:end] = best
    return out


def js_divergence(a: Iterable[Any], b: Iterable[Any], bins: int = 30, value_range: tuple[float, float] | None = None) -> float | None:
    x = finite(a)
    y = finite(b)
    if len(x) == 0 or len(y) == 0:
        return None
    if value_range is None:
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        if hi <= lo:
            hi = lo + 1.0
        value_range = (lo, hi)
    hx, _ = np.histogram(x, bins=bins, range=value_range, density=False)
    hy, _ = np.histogram(y, bins=bins, range=value_range, density=False)
    px = hx.astype(np.float64) + 1e-12
    py = hy.astype(np.float64) + 1e-12
    px /= px.sum()
    py /= py.sum()
    m = 0.5 * (px + py)
    return float(0.5 * np.sum(px * np.log(px / m)) + 0.5 * np.sum(py * np.log(py / m)))


def pearson(x_values: Iterable[Any], y_values: Iterable[Any]) -> float | None:
    pairs = []
    for x, y in zip(x_values, y_values):
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    if np.std(arr[:, 0]) == 0 or np.std(arr[:, 1]) == 0:
        return None
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def write_markdown_table(path: str | Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        if not rows:
            fh.write("No rows.\n")
            return
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col)
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                elif val is None:
                    vals.append("NA")
                else:
                    vals.append(str(val))
            fh.write("| " + " | ".join(vals) + " |\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})

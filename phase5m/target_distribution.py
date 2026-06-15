from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser()) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _sample_indices(values: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.size <= int(max_items):
        return values
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(values, size=int(max_items), replace=False), dtype=np.int64)


def summarize_bars_target_indices(
    option_edges_csv: str | Path,
    edge_segments_npz: str | Path,
    *,
    max_targets: int = 50000,
    seed: int = 0,
) -> pd.DataFrame:
    edges = pd.read_csv(Path(option_edges_csv).expanduser())
    segments = _load_npz(edge_segments_npz)
    required = {"edge_id", "global_i", "global_j"}
    missing = required - set(segments)
    if missing:
        raise KeyError(f"edge_segments missing required arrays: {sorted(missing)}")
    edge_ids = np.asarray(segments["edge_id"], dtype=np.int64)
    global_i = np.asarray(segments["global_i"], dtype=np.int64)
    global_j = np.asarray(segments["global_j"], dtype=np.int64)
    sampled = _sample_indices(np.arange(edge_ids.shape[0], dtype=np.int64), max_targets, seed)
    df = pd.DataFrame(
        {
            "segment_index": sampled,
            "edge_id": edge_ids[sampled],
            "global_i": global_i[sampled],
            "global_j": global_j[sampled],
        }
    )
    meta_cols = [
        "edge_id",
        "src",
        "dst",
        "num_segments",
        "num_unique_starts",
        "num_unique_episodes",
        "median_h",
        "edge_bottleneck_score",
    ]
    return df.merge(edges[[c for c in meta_cols if c in edges.columns]], on="edge_id", how="left")


def nearest_distance_summary(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("query and reference must be rank-2 arrays")
    if query.shape[1] != reference.shape[1]:
        raise ValueError(f"dimension mismatch: query {query.shape}, reference {reference.shape}")
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return {
            "num_queries": int(query.shape[0]),
            "num_reference": int(reference.shape[0]),
            "mean_nn_l2": None,
            "median_nn_l2": None,
            "p90_nn_l2": None,
            "p95_nn_l2": None,
            "max_nn_l2": None,
        }
    mins = np.empty(query.shape[0], dtype=np.float32)
    ref_norm = np.sum(reference * reference, axis=1, dtype=np.float32)
    for start in range(0, query.shape[0], int(chunk_size)):
        q = query[start : start + int(chunk_size)]
        q_norm = np.sum(q * q, axis=1, dtype=np.float32)[:, None]
        dist2 = np.maximum(q_norm + ref_norm[None, :] - 2.0 * (q @ reference.T), 0.0)
        mins[start : start + q.shape[0]] = np.sqrt(np.min(dist2, axis=1))
    return {
        "num_queries": int(query.shape[0]),
        "num_reference": int(reference.shape[0]),
        "mean_nn_l2": float(np.mean(mins)),
        "median_nn_l2": float(np.median(mins)),
        "p90_nn_l2": float(np.quantile(mins, 0.90)),
        "p95_nn_l2": float(np.quantile(mins, 0.95)),
        "max_nn_l2": float(np.max(mins)),
    }


def run_target_distribution_audit(
    *,
    option_edges_csv: str | Path,
    edge_segments_npz: str | Path,
    gas_dataset_embeddings_path: str | Path,
    output_dir: str | Path,
    max_targets: int = 50000,
    max_reference: int = 200000,
    seed: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    embeddings_path = Path(gas_dataset_embeddings_path).expanduser()
    target_df = summarize_bars_target_indices(
        option_edges_csv,
        edge_segments_npz,
        max_targets=max_targets,
        seed=seed,
    )
    target_df.to_csv(out / "bars_target_sample.csv", index=False)
    if not embeddings_path.exists():
        summary = {
            "status": "blocked_missing_gas_dataset_embeddings",
            "gas_dataset_embeddings_path": str(embeddings_path),
            "num_bars_target_samples": int(target_df.shape[0]),
            "note": "Cannot assess GAS-policy target compatibility without GAS TDR dataset embeddings or live GAS get_phi.",
        }
        _write_json(out / "target_distribution_audit_summary.json", summary)
        pd.DataFrame([summary]).to_csv(out / "target_distribution_audit_summary.csv", index=False)
        return {"summary": summary, "target_sample": target_df, "output_dir": out}

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected rank-2 GAS embeddings, got {embeddings.shape}")
    unique_targets = np.asarray(sorted(set(int(x) for x in target_df["global_j"].to_numpy(dtype=np.int64))), dtype=np.int64)
    valid_targets = unique_targets[(unique_targets >= 0) & (unique_targets < embeddings.shape[0])]
    sampled_ref_idx = _sample_indices(np.arange(embeddings.shape[0], dtype=np.int64), max_reference, seed + 17)
    query = np.asarray(embeddings[valid_targets], dtype=np.float32)
    reference = np.asarray(embeddings[sampled_ref_idx], dtype=np.float32)
    nn_summary = nearest_distance_summary(query, reference)
    summary = {
        "status": "completed_embedding_space_proxy",
        "gas_dataset_embeddings_path": str(embeddings_path),
        "num_bars_target_samples": int(target_df.shape[0]),
        "num_unique_bars_terminations": int(unique_targets.shape[0]),
        "num_valid_embedding_targets": int(valid_targets.shape[0]),
        "num_reference_embeddings": int(reference.shape[0]),
        **{f"target_to_dataset_{k}": v for k, v in nn_summary.items()},
        "note": (
            "This is only an embedding-space proxy. It checks whether BARS target "
            "states are near GAS dataset embeddings, not whether GAS actor can "
            "execute the corresponding skills from arbitrary current states."
        ),
    }
    pd.DataFrame({"global_j": valid_targets}).to_csv(out / "valid_bars_target_indices.csv", index=False)
    _write_json(out / "target_distribution_audit_summary.json", summary)
    pd.DataFrame([summary]).to_csv(out / "target_distribution_audit_summary.csv", index=False)
    return {"summary": summary, "target_sample": target_df, "output_dir": out}


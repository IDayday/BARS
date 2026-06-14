from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_norm(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape[0], dtype=np.float64)
    lo = float(np.nanpercentile(finite, 5))
    hi = float(np.nanpercentile(finite, 95))
    if hi <= lo:
        return np.zeros(arr.shape[0], dtype=np.float64)
    out = (np.nan_to_num(arr, nan=hi) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def compute_edge_distance_proxy(
    dataset: dict[str, Any],
    edge_segments: dict[str, np.ndarray],
    edge_ids: np.ndarray,
    max_segments_per_edge: int = 256,
    seed: int = 0,
) -> pd.DataFrame:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    edge_ids_all = np.asarray(edge_segments["edge_id"], dtype=np.int64)
    global_i = np.asarray(edge_segments["global_i"], dtype=np.int64)
    global_j = np.asarray(edge_segments["global_j"], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rows = []
    for edge_id in np.asarray(edge_ids, dtype=np.int64):
        idx = np.flatnonzero(edge_ids_all == int(edge_id))
        if idx.size == 0:
            rows.append({"edge_id": int(edge_id), "obs_goal_knn_distance": np.nan})
            continue
        if idx.size > int(max_segments_per_edge):
            idx = rng.choice(idx, size=int(max_segments_per_edge), replace=False)
        starts = observations[global_i[idx]]
        goals = observations[global_j[idx]]
        dist = np.linalg.norm(starts.reshape(starts.shape[0], -1) - goals.reshape(goals.shape[0], -1), axis=1)
        rows.append({"edge_id": int(edge_id), "obs_goal_knn_distance": float(np.mean(dist))})
    return pd.DataFrame(rows)


def compute_compatibility_proxy(
    pair_compatibility: pd.DataFrame | None,
    edge_ids: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame({"edge_id": np.asarray(edge_ids, dtype=np.int64)})
    if pair_compatibility is None or pair_compatibility.empty:
        out["outgoing_mean_termination_bridge_coverage"] = 0.0
        out["incoming_mean_termination_bridge_coverage"] = 0.0
        out["outgoing_incompatible_fraction"] = 1.0
        out["incoming_incompatible_fraction"] = 1.0
        return out
    coverage_col = (
        "termination_bridge_coverage"
        if "termination_bridge_coverage" in pair_compatibility.columns
        else "strict_compatible"
    )
    strict = pd.to_numeric(pair_compatibility.get("strict_compatible", False), errors="coerce").fillna(0.0)
    tmp = pair_compatibility.copy()
    tmp["_incompatible"] = 1.0 - strict.astype(float)
    outgoing = (
        tmp.groupby("edge_id_first", sort=False)
        .agg(
            outgoing_mean_termination_bridge_coverage=(coverage_col, "mean"),
            outgoing_incompatible_fraction=("_incompatible", "mean"),
        )
        .reset_index()
        .rename(columns={"edge_id_first": "edge_id"})
    )
    incoming = (
        tmp.groupby("edge_id_second", sort=False)
        .agg(
            incoming_mean_termination_bridge_coverage=(coverage_col, "mean"),
            incoming_incompatible_fraction=("_incompatible", "mean"),
        )
        .reset_index()
        .rename(columns={"edge_id_second": "edge_id"})
    )
    out = out.merge(outgoing, on="edge_id", how="left").merge(incoming, on="edge_id", how="left")
    for col, default in [
        ("outgoing_mean_termination_bridge_coverage", 0.0),
        ("incoming_mean_termination_bridge_coverage", 0.0),
        ("outgoing_incompatible_fraction", 1.0),
        ("incoming_incompatible_fraction", 1.0),
    ]:
        out[col] = out[col].fillna(default).astype(float)
    return out


def add_ood_proxy(cert: pd.DataFrame) -> pd.DataFrame:
    out = cert.copy()
    support_col = "num_unique_starts" if "num_unique_starts" in out.columns else "num_segments"
    support = pd.to_numeric(out.get(support_col, 0), errors="coerce").fillna(0)
    support_thr = float(support.median()) if not support.empty else 0.0
    out["low_support_flag"] = (support < support_thr).astype(float)
    out["action_residual_score"] = _safe_norm(out.get("edge_action_mse", pd.Series(np.zeros(out.shape[0]))))
    out["obs_goal_knn_distance_norm"] = _safe_norm(out.get("obs_goal_knn_distance", pd.Series(np.zeros(out.shape[0]))))
    out["edge_ood_score"] = (
        out["action_residual_score"] + out["obs_goal_knn_distance_norm"] + out["low_support_flag"]
    ) / 3.0
    return out


def compute_proxy_scores(
    option_edges: pd.DataFrame,
    heldout_support: pd.DataFrame,
    policy_scores: pd.DataFrame,
    distance_proxy: pd.DataFrame,
    compatibility_proxy: pd.DataFrame,
    weights: dict[str, float] | None = None,
    threshold: float = 0.35,
    min_heldout_support_lcb: float = 0.01,
) -> pd.DataFrame:
    weights = weights or {"support": 0.4, "policy": 0.3, "compat": 0.2, "ood": 0.1}
    base_cols = [
        "edge_id",
        "src",
        "dst",
        "edge_bottleneck_score",
        "num_unique_starts",
        "num_unique_episodes",
        "num_episodes",
        "median_h",
        "num_segments",
    ]
    out = option_edges[[col for col in base_cols if col in option_edges.columns]].copy()
    if "num_unique_episodes" not in out.columns and "num_episodes" in out.columns:
        out["num_unique_episodes"] = out["num_episodes"]
    for frame in [heldout_support, policy_scores, distance_proxy, compatibility_proxy]:
        out = out.merge(frame.drop(columns=[c for c in ["src", "dst"] if c in frame.columns]), on="edge_id", how="left")
    out["heldout_support_lcb"] = out["heldout_support_lcb"].fillna(0.0)
    out["edge_policy_support_score"] = out["edge_policy_support_score"].fillna(0.0)
    out["outgoing_mean_termination_bridge_coverage"] = out[
        "outgoing_mean_termination_bridge_coverage"
    ].fillna(0.0)
    out["incoming_mean_termination_bridge_coverage"] = out[
        "incoming_mean_termination_bridge_coverage"
    ].fillna(0.0)
    out = add_ood_proxy(out)
    compat = out[
        ["outgoing_mean_termination_bridge_coverage", "incoming_mean_termination_bridge_coverage"]
    ].mean(axis=1)
    out["edge_proxy_score"] = (
        float(weights.get("support", 0.4)) * out["heldout_support_lcb"]
        + float(weights.get("policy", 0.3)) * out["edge_policy_support_score"]
        + float(weights.get("compat", 0.2)) * compat
        - float(weights.get("ood", 0.1)) * out["edge_ood_score"]
    )
    out["certified_offline_binary"] = (
        (out["edge_proxy_score"] >= float(threshold))
        & (out["heldout_support_lcb"] >= float(min_heldout_support_lcb))
    )
    ordered = [
        "edge_id",
        "src",
        "dst",
        "heldout_support_lcb",
        "edge_action_mse",
        "edge_action_mse_ucb",
        "edge_policy_support_score",
        "obs_goal_knn_distance",
        "action_residual_score",
        "low_support_flag",
        "edge_ood_score",
        "outgoing_mean_termination_bridge_coverage",
        "incoming_mean_termination_bridge_coverage",
        "outgoing_incompatible_fraction",
        "incoming_incompatible_fraction",
        "edge_proxy_score",
        "certified_offline_binary",
        "edge_bottleneck_score",
        "num_unique_starts",
        "num_unique_episodes",
        "median_h",
        "num_segments",
    ]
    extras = [col for col in out.columns if col not in ordered]
    return out[[col for col in ordered if col in out.columns] + extras].sort_values("edge_proxy_score", ascending=False)


def certification_summary(cert: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if cert.empty:
        return {
            "num_edges": 0,
            "certified_offline_edges": 0,
            "certified_offline_rate": 0.0,
            "note": "Reset-free offline proxy; not rollout success.",
            "config": config,
        }
    return {
        "num_edges": int(cert.shape[0]),
        "certified_offline_edges": int(cert["certified_offline_binary"].sum()),
        "certified_offline_rate": float(cert["certified_offline_binary"].mean()),
        "mean_edge_proxy_score": float(cert["edge_proxy_score"].mean()),
        "median_edge_proxy_score": float(cert["edge_proxy_score"].median()),
        "mean_heldout_support_lcb": float(cert["heldout_support_lcb"].mean()),
        "mean_edge_action_mse": float(pd.to_numeric(cert["edge_action_mse"], errors="coerce").mean()),
        "mean_edge_policy_support_score": float(cert["edge_policy_support_score"].mean()),
        "mean_edge_ood_score": float(cert["edge_ood_score"].mean()),
        "note": "Reset-free offline proxy; not rollout success.",
        "config": config,
    }

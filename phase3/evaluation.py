from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase1.diagnostics import build_knn_edges
from phase3.edge_rollout import evaluate_edge_rollouts


OFFLINE_MSE_NOTE = "Offline action MSE is not edge execution success."


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def phase2_method_name(phase2_run_dir: str | Path) -> str:
    name = Path(phase2_run_dir).name
    if "_budget" in name:
        return name.split("_budget", 1)[0]
    return name


def default_phase3_output_dir(dataset_name: str, phase2_run_dir: str | Path, root: str | Path = "results/phase3") -> Path:
    return Path(root) / _dataset_key(dataset_name) / phase2_method_name(phase2_run_dir)


def load_phase2_artifacts(phase2_run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    run_dir = Path(phase2_run_dir).expanduser()
    option_edges = pd.read_csv(run_dir / "option_edges.csv")
    selected_nodes = pd.read_csv(run_dir / "selected_nodes.csv")
    with np.load(run_dir / "edge_segments.npz") as data:
        edge_segments = {key: np.asarray(data[key]) for key in data.files}
    return option_edges, edge_segments, selected_nodes


def _final_metric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return float("nan")
    return float(df.iloc[-1][column])


def _best_metric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else float("nan")


def _edge_metric_mean(df: pd.DataFrame) -> float:
    if df.empty or "val_action_mse" not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df["val_action_mse"], errors="coerce").mean())


def _edge_sample_sum(df: pd.DataFrame) -> int:
    if df.empty or "num_val_samples" not in df.columns:
        return 0
    return int(pd.to_numeric(df["num_val_samples"], errors="coerce").fillna(0).sum())


def _median_threshold(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _group_row(name: str, edge_metrics: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict[str, Any]:
    sub = edge_metrics.loc[np.asarray(mask, dtype=bool)].copy()
    return {
        "group": name,
        "num_edges": int(sub.shape[0]),
        "num_val_samples": _edge_sample_sum(sub),
        "val_action_mse": _edge_metric_mean(sub),
    }


def attach_edge_compatibility_context(
    edge_metrics: pd.DataFrame,
    pair_compatibility: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach edge-level compatibility context derived from adjacent edge pairs."""

    out = edge_metrics.copy()
    if pair_compatibility is None or pair_compatibility.empty:
        return out
    compat_col = (
        "termination_bridge_coverage"
        if "termination_bridge_coverage" in pair_compatibility.columns
        else "strict_compatible"
    )
    outgoing = (
        pair_compatibility.groupby("edge_id_first", sort=False)[compat_col]
        .mean()
        .rename("outgoing_mean_termination_bridge_coverage")
        .reset_index()
        .rename(columns={"edge_id_first": "edge_id"})
    )
    incoming = (
        pair_compatibility.groupby("edge_id_second", sort=False)[compat_col]
        .mean()
        .rename("incoming_mean_termination_bridge_coverage")
        .reset_index()
        .rename(columns={"edge_id_second": "edge_id"})
    )
    out = out.merge(outgoing, on="edge_id", how="left")
    out = out.merge(incoming, on="edge_id", how="left")
    out["strict_compatible_context"] = out[
        ["outgoing_mean_termination_bridge_coverage", "incoming_mean_termination_bridge_coverage"]
    ].mean(axis=1, skipna=True)
    return out


def normalize_edge_val_metrics(
    edge_metrics: pd.DataFrame,
    option_edges: pd.DataFrame,
    pair_compatibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return edge validation metrics with stable Phase 3A column names."""

    metrics = edge_metrics.copy()
    if "num_val_samples" not in metrics.columns and "num_examples" in metrics.columns:
        metrics = metrics.rename(columns={"num_examples": "num_val_samples"})
    if "num_val_samples" not in metrics.columns:
        metrics["num_val_samples"] = 0
    option_cols = [
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
    existing = [col for col in option_cols if col in option_edges.columns]
    metadata = option_edges[existing].copy()
    duplicate_meta = [col for col in existing if col != "edge_id" and col in metrics.columns]
    if duplicate_meta:
        metrics = metrics.drop(columns=duplicate_meta)
    merged = metrics.merge(metadata, on="edge_id", how="left") if "edge_id" in metrics.columns else metrics
    if "num_unique_episodes" not in merged.columns and "num_episodes" in merged.columns:
        merged["num_unique_episodes"] = merged["num_episodes"]
    merged = attach_edge_compatibility_context(merged, pair_compatibility)
    ordered = [
        "edge_id",
        "src",
        "dst",
        "val_action_mse",
        "num_val_samples",
        "edge_bottleneck_score",
        "num_unique_starts",
        "num_unique_episodes",
        "median_h",
        "num_segments",
        "strict_compatible_context",
    ]
    extras = [col for col in merged.columns if col not in ordered]
    return merged[[col for col in ordered if col in merged.columns] + extras].sort_values(
        "edge_id", kind="mergesort"
    )


def compute_grouped_edge_val_metrics(edge_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Summarize held-out action MSE by Phase 2 edge metadata groups."""

    if edge_metrics.empty:
        empty = pd.DataFrame(columns=["group", "num_edges", "num_val_samples", "val_action_mse"])
        return empty, {
            "bottleneck_edge_val_mse": float("nan"),
            "non_bottleneck_edge_val_mse": float("nan"),
            "high_support_edge_val_mse": float("nan"),
            "low_support_edge_val_mse": float("nan"),
            "short_horizon_edge_val_mse": float("nan"),
            "long_horizon_edge_val_mse": float("nan"),
        }

    support_col = "num_unique_starts" if "num_unique_starts" in edge_metrics.columns else "num_segments"
    bottleneck_thr = _median_threshold(edge_metrics, "edge_bottleneck_score")
    support_thr = _median_threshold(edge_metrics, support_col)
    horizon_thr = _median_threshold(edge_metrics, "median_h")
    compat_col = (
        "strict_compatible_context"
        if "strict_compatible_context" in edge_metrics.columns
        else (
            "outgoing_mean_termination_bridge_coverage"
            if "outgoing_mean_termination_bridge_coverage" in edge_metrics.columns
            else None
        )
    )
    compat_thr = _median_threshold(edge_metrics, compat_col) if compat_col is not None else None

    rows: list[dict[str, Any]] = [
        _group_row("all_edges", edge_metrics, np.ones(edge_metrics.shape[0], dtype=bool))
    ]
    if bottleneck_thr is not None:
        rows.append(_group_row("high_bottleneck_edges", edge_metrics, edge_metrics["edge_bottleneck_score"] >= bottleneck_thr))
        rows.append(_group_row("low_bottleneck_edges", edge_metrics, edge_metrics["edge_bottleneck_score"] < bottleneck_thr))
    if support_thr is not None:
        rows.append(_group_row("high_support_edges", edge_metrics, edge_metrics[support_col] >= support_thr))
        rows.append(_group_row("low_support_edges", edge_metrics, edge_metrics[support_col] < support_thr))
    if horizon_thr is not None:
        rows.append(_group_row("short_horizon_edges", edge_metrics, edge_metrics["median_h"] <= horizon_thr))
        rows.append(_group_row("long_horizon_edges", edge_metrics, edge_metrics["median_h"] > horizon_thr))
    if compat_col is not None and compat_thr is not None:
        values = pd.to_numeric(edge_metrics[compat_col], errors="coerce").fillna(-np.inf)
        rows.append(_group_row("high_compatibility_edges", edge_metrics, values >= compat_thr))
        rows.append(_group_row("low_compatibility_edges", edge_metrics, values < compat_thr))

    grouped = pd.DataFrame(rows)
    lookup = grouped.set_index("group")["val_action_mse"].to_dict() if not grouped.empty else {}
    summary = {
        "bottleneck_edge_val_mse": float(lookup.get("high_bottleneck_edges", np.nan)),
        "non_bottleneck_edge_val_mse": float(lookup.get("low_bottleneck_edges", np.nan)),
        "high_support_edge_val_mse": float(lookup.get("high_support_edges", np.nan)),
        "low_support_edge_val_mse": float(lookup.get("low_support_edges", np.nan)),
        "short_horizon_edge_val_mse": float(lookup.get("short_horizon_edges", np.nan)),
        "long_horizon_edge_val_mse": float(lookup.get("long_horizon_edges", np.nan)),
    }
    return grouped, summary


def summarize_supervised_metrics(
    train_metrics: pd.DataFrame,
    val_metrics: pd.DataFrame,
    edge_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped, grouped_summary = compute_grouped_edge_val_metrics(edge_metrics)
    summary = {
        "final_train_action_mse": _final_metric(train_metrics, "train_action_mse"),
        "final_val_action_mse": _final_metric(val_metrics, "val_action_mse"),
        "best_val_action_mse": _best_metric(val_metrics, "val_action_mse"),
        **grouped_summary,
        "note": OFFLINE_MSE_NOTE,
    }
    return grouped, summary


def synthetic_edge_segments_for_cluster_edges(
    edges: pd.DataFrame,
    labels: np.ndarray,
    samples_per_edge: int,
    H: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(seed)
    edge_ids: list[int] = []
    ep_ids: list[int] = []
    ts: list[int] = []
    hs: list[int] = []
    gis: list[int] = []
    gjs: list[int] = []
    for row in edges.itertuples(index=False):
        starts = np.flatnonzero(labels == int(row.src))
        goals = np.flatnonzero(labels == int(row.dst))
        if starts.size == 0 or goals.size == 0:
            continue
        for _ in range(int(samples_per_edge)):
            gi = int(rng.choice(starts))
            gj = int(rng.choice(goals))
            edge_ids.append(int(row.edge_id))
            ep_ids.append(-1)
            ts.append(gi)
            hs.append(max(1, int(H)))
            gis.append(gi)
            gjs.append(gj)
    return {
        "edge_id": np.asarray(edge_ids, dtype=np.int64),
        "ep_id": np.asarray(ep_ids, dtype=np.int64),
        "t": np.asarray(ts, dtype=np.int64),
        "h": np.asarray(hs, dtype=np.int64),
        "global_i": np.asarray(gis, dtype=np.int64),
        "global_j": np.asarray(gjs, dtype=np.int64),
    }


def _edge_frame(
    edge_pairs: list[tuple[int, int]],
    start_edge_id: int,
    source: str,
    option_edges: pd.DataFrame,
) -> pd.DataFrame:
    support_meta = option_edges.set_index(["src", "dst"], drop=False) if not option_edges.empty else pd.DataFrame()
    rows = []
    for offset, (src, dst) in enumerate(edge_pairs):
        meta = support_meta.loc[(src, dst)] if not support_meta.empty and (src, dst) in support_meta.index else None
        rows.append(
            {
                "edge_id": int(start_edge_id + offset),
                "src": int(src),
                "dst": int(dst),
                "edge_source": source,
                "num_segments": int(getattr(meta, "num_segments", 0)) if meta is not None else 0,
                "num_episodes": int(getattr(meta, "num_episodes", 0)) if meta is not None else 0,
                "num_unique_starts": int(getattr(meta, "num_unique_starts", 0)) if meta is not None else 0,
                "num_unique_episodes": int(getattr(meta, "num_unique_episodes", 0)) if meta is not None else 0,
                "median_h": float(getattr(meta, "median_h", 1.0)) if meta is not None else 1.0,
                "max_h": float(getattr(meta, "max_h", 1.0)) if meta is not None else 1.0,
                "edge_bottleneck_score": float(getattr(meta, "edge_bottleneck_score", 0.0)) if meta is not None else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_baseline_edge_sets(
    option_edges: pd.DataFrame,
    selected_nodes: pd.DataFrame,
    observations: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    edge_budget: int | None = None,
    knn_k: int = 10,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    selected = set(selected_nodes[selected_nodes["selected"]]["cluster"].astype(int).tolist())
    support_pairs = {(int(row.src), int(row.dst)) for row in option_edges.itertuples(index=False)}
    edge_budget = int(option_edges.shape[0] if edge_budget is None else edge_budget)
    rng = np.random.default_rng(seed)

    knn_all = sorted(
        (int(src), int(dst))
        for src, dst in build_knn_edges(observations, labels, n_clusters, max(1, int(knn_k)), "cluster_center_knn", seed=seed)
        if int(src) in selected and int(dst) in selected and int(src) != int(dst)
    )
    knn_pairs = knn_all[:edge_budget]
    unsupported_knn = [pair for pair in knn_all if pair not in support_pairs][:edge_budget]

    possible = sorted((int(src), int(dst)) for src in selected for dst in selected if int(src) != int(dst))
    if possible and edge_budget > 0:
        take = min(edge_budget, len(possible))
        random_pairs = [possible[int(i)] for i in rng.choice(len(possible), size=take, replace=False)]
    else:
        random_pairs = []

    return {
        "support_option_edges": option_edges.copy(),
        "kNN_same_nodes": _edge_frame(knn_pairs, 1_000_000, "kNN_same_nodes", option_edges),
        "random_same_edge_budget": _edge_frame(random_pairs, 2_000_000, "random_same_edge_budget", option_edges),
        "unsupported_kNN_edges": _edge_frame(unsupported_knn, 3_000_000, "unsupported_kNN_edges", option_edges),
    }


def unsupported_rate_against_support(edge_df: pd.DataFrame, option_edges: pd.DataFrame) -> float:
    if edge_df.empty:
        return 0.0
    support_pairs = {(int(row.src), int(row.dst)) for row in option_edges.itertuples(index=False)}
    unsupported = [(int(row.src), int(row.dst)) not in support_pairs for row in edge_df.itertuples(index=False)]
    return float(np.mean(unsupported)) if unsupported else 0.0


def summarize_baseline_execution(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        [
            "edge_source",
            "unsupported_edge_rate",
            "num_eval_edges",
            "mean_success_rate",
            "median_success_rate",
        ]
    ]


def evaluate_baseline_edge_sets(
    env: Any,
    policy: Any,
    dataset: dict[str, Any],
    baseline_edges: dict[str, pd.DataFrame],
    support_option_edges: pd.DataFrame,
    support_edge_segments: dict[str, np.ndarray],
    labels: np.ndarray,
    cluster_model: dict[str, Any],
    starts_per_edge: int,
    H: int,
    num_edges: int,
    success_mode: str,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_edge_metrics = []
    summary_rows = []
    for idx, (source, edges) in enumerate(baseline_edges.items()):
        if source == "support_option_edges":
            segments = support_edge_segments
            eval_edges = edges
        else:
            eval_edges = edges.copy()
            segments = synthetic_edge_segments_for_cluster_edges(
                eval_edges,
                labels,
                samples_per_edge=starts_per_edge,
                H=H,
                seed=seed + idx * 101,
            )
        metrics, _ = evaluate_edge_rollouts(
            env,
            policy,
            dataset,
            eval_edges,
            segments,
            cluster_model,
            num_edges=num_edges,
            starts_per_edge=starts_per_edge,
            horizon_mode="fixed_H" if source != "support_option_edges" else "edge_median",
            success_mode=success_mode,
            fixed_H=H,
            seed=seed + idx * 17,
        )
        if not metrics.empty:
            metrics = metrics.copy()
            metrics["edge_source"] = source
            all_edge_metrics.append(metrics)
        summary_rows.append(
            {
                "edge_source": source,
                "unsupported_edge_rate": 0.0
                if source == "support_option_edges"
                else unsupported_rate_against_support(eval_edges, support_option_edges),
                "num_eval_edges": int(metrics.shape[0]),
                "mean_success_rate": float(metrics["success_rate"].mean()) if not metrics.empty else 0.0,
                "median_success_rate": float(metrics["success_rate"].median()) if not metrics.empty else 0.0,
            }
        )
    edge_metrics = pd.concat(all_edge_metrics, ignore_index=True) if all_edge_metrics else pd.DataFrame()
    return edge_metrics, summarize_baseline_execution(summary_rows)

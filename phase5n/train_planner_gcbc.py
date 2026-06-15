from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phase3.models import GCBCMLP, action_mse
from phase3.plotting import plot_training_curves
from phase3.train_gcbc import _json_safe, write_json
from phase5n.planner_subgoal_dataset import (
    TARGET_ID_TO_SOURCE,
    PlannerMixedGCBCDataset,
    build_planner_relevant_edge_weights,
    load_npz_arrays,
)


def _make_loader(dataset: PlannerMixedGCBCDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, generator=generator)


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _weighted_action_mse(pred: torch.Tensor, action: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    per_sample = torch.mean((pred - action.float()) ** 2, dim=1)
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).clamp(min=1e-8)
    return torch.sum(per_sample * weights) / torch.sum(weights)


def evaluate_mixed_policy_mse(
    model: GCBCMLP,
    dataset: PlannerMixedGCBCDataset,
    *,
    batch_size: int,
    max_examples: int,
    device: torch.device,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    if len(dataset) == 0 or int(max_examples) <= 0:
        return 0.0, pd.DataFrame(), pd.DataFrame()
    eval_ds = dataset.with_indices(max_examples=int(max_examples), seed=dataset.seed + 991)
    loader = _make_loader(eval_ds, batch_size=batch_size, shuffle=False, seed=dataset.seed + 13)
    model.eval()
    total = 0.0
    count = 0
    source_sum: dict[int, float] = {}
    source_count: dict[int, int] = {}
    edge_sum: dict[int, float] = {}
    edge_count: dict[int, int] = {}
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            model_edge_ids = batch["edge_id"].long().clamp(min=0)
            pred = model(batch["obs"], batch["goal"], batch["remaining_h"], model_edge_ids)
            per_sample = torch.mean((pred - batch["action"].float()) ** 2, dim=1).detach().cpu().numpy()
            source_ids = batch["target_source_id"].detach().cpu().numpy()
            edge_ids = batch["edge_id"].detach().cpu().numpy()
            total += float(per_sample.sum())
            count += int(per_sample.shape[0])
            for source_id, edge_id, value in zip(source_ids, edge_ids, per_sample):
                sid = int(source_id)
                eid = int(edge_id)
                source_sum[sid] = source_sum.get(sid, 0.0) + float(value)
                source_count[sid] = source_count.get(sid, 0) + 1
                if eid >= 0:
                    edge_sum[eid] = edge_sum.get(eid, 0.0) + float(value)
                    edge_count[eid] = edge_count.get(eid, 0) + 1
            if count >= int(max_examples):
                break
    source_rows = [
        {
            "target_source_id": int(source_id),
            "target_source": TARGET_ID_TO_SOURCE.get(int(source_id), str(source_id)),
            "val_action_mse": float(source_sum[source_id] / max(1, source_count[source_id])),
            "num_examples": int(source_count[source_id]),
        }
        for source_id in sorted(source_sum)
    ]
    edge_rows = [
        {
            "edge_id": int(edge_id),
            "val_action_mse": float(edge_sum[edge_id] / max(1, edge_count[edge_id])),
            "num_examples": int(edge_count[edge_id]),
        }
        for edge_id in sorted(edge_sum)
    ]
    return float(total / max(1, count)), pd.DataFrame(source_rows), pd.DataFrame(edge_rows)


def summarize_source_metrics(source_metrics: pd.DataFrame) -> dict[str, float]:
    out = {
        "final_goal_hindsight_val_mse": np.nan,
        "support_edge_local_val_mse": np.nan,
        "planner_first_edge_replay_val_mse": np.nan,
    }
    if source_metrics.empty:
        return out
    for row in source_metrics.itertuples(index=False):
        key = f"{getattr(row, 'target_source')}_val_mse"
        out[key] = float(getattr(row, "val_action_mse"))
    return out


def summarize_edge_groups(edge_metrics: pd.DataFrame, option_edges: pd.DataFrame) -> dict[str, float]:
    if edge_metrics.empty:
        return {
            "bottleneck_edge_val_mse": np.nan,
            "non_bottleneck_edge_val_mse": np.nan,
            "high_support_edge_val_mse": np.nan,
            "low_support_edge_val_mse": np.nan,
            "short_horizon_edge_val_mse": np.nan,
            "long_horizon_edge_val_mse": np.nan,
            "planner_used_edge_val_mse": np.nan,
            "not_planner_used_edge_val_mse": np.nan,
        }
    merged = edge_metrics.merge(option_edges, on="edge_id", how="left")
    support_col = "num_unique_starts" if "num_unique_starts" in merged.columns else "num_segments"
    bottleneck_thr = float(pd.to_numeric(merged.get("edge_bottleneck_score", 0.0), errors="coerce").median())
    support_thr = float(pd.to_numeric(merged.get(support_col, 0.0), errors="coerce").median())
    horizon_thr = float(pd.to_numeric(merged.get("median_h", 0.0), errors="coerce").median())

    def _mean(mask: pd.Series) -> float:
        sub = merged[mask]
        return float(pd.to_numeric(sub["val_action_mse"], errors="coerce").mean()) if not sub.empty else np.nan

    planner_used = pd.to_numeric(merged.get("planner_first_edge_count", 0.0), errors="coerce").fillna(0.0) > 0.0
    bottleneck = pd.to_numeric(merged.get("edge_bottleneck_score", 0.0), errors="coerce").fillna(0.0)
    support = pd.to_numeric(merged.get(support_col, 0.0), errors="coerce").fillna(0.0)
    horizon = pd.to_numeric(merged.get("median_h", 0.0), errors="coerce").fillna(0.0)
    return {
        "bottleneck_edge_val_mse": _mean(bottleneck >= bottleneck_thr),
        "non_bottleneck_edge_val_mse": _mean(bottleneck < bottleneck_thr),
        "high_support_edge_val_mse": _mean(support >= support_thr),
        "low_support_edge_val_mse": _mean(support < support_thr),
        "short_horizon_edge_val_mse": _mean(horizon <= horizon_thr),
        "long_horizon_edge_val_mse": _mean(horizon > horizon_thr),
        "planner_used_edge_val_mse": _mean(planner_used),
        "not_planner_used_edge_val_mse": _mean(~planner_used),
    }


def train_planner_subgoal_gcbc(
    *,
    dataset: dict[str, Any],
    option_edges_csv: str | Path | pd.DataFrame,
    edge_segments_npz: str | Path | dict[str, np.ndarray],
    output_dir: str | Path,
    batch_size: int = 1024,
    num_steps: int = 100000,
    lr: float = 3e-4,
    hidden_dims: list[int] | str | None = None,
    seed: int = 0,
    val_fraction: float = 0.05,
    val_examples: int = 8192,
    train_examples: int | None = None,
    log_interval: int | None = None,
    device: str | None = None,
    use_remaining_h: bool = True,
    edge_embedding_dim: int = 0,
    source_probabilities: dict[str, float] | None = None,
    source_loss_weights: dict[str, float] | None = None,
    num_planner_queries: int = 5000,
    base_loss_weight_mode: str = "support_bottleneck",
    base_loss_weight_strength: float = 0.3,
    planner_usage_strength: float = 0.35,
    planner_first_edge_strength: float = 0.65,
    min_loss_weight: float = 0.5,
    max_loss_weight: float = 2.5,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    option_edges = option_edges_csv.copy() if isinstance(option_edges_csv, pd.DataFrame) else pd.read_csv(Path(option_edges_csv).expanduser())
    edge_segments = (
        {key: np.asarray(value) for key, value in edge_segments_npz.items()}
        if isinstance(edge_segments_npz, dict)
        else load_npz_arrays(edge_segments_npz)
    )
    planner_weights, path_rows = build_planner_relevant_edge_weights(
        option_edges,
        num_queries=int(num_planner_queries),
        seed=int(seed),
        base_loss_weight_mode=base_loss_weight_mode,
        base_loss_weight_strength=float(base_loss_weight_strength),
        planner_usage_strength=float(planner_usage_strength),
        planner_first_edge_strength=float(planner_first_edge_strength),
        min_weight=float(min_loss_weight),
        max_weight=float(max_loss_weight),
    )
    planner_weights.to_csv(out / "planner_edge_weights.csv", index=False)
    path_rows.to_csv(out / "planner_query_paths.csv", index=False)

    train_len = train_examples
    if train_len is None:
        train_len = max(int(batch_size) * max(20, min(int(num_steps), 1000)), int(batch_size))
    full = PlannerMixedGCBCDataset(
        dataset,
        option_edges,
        edge_segments,
        planner_weights,
        max_examples=int(train_len),
        source_probabilities=source_probabilities,
        source_loss_weights=source_loss_weights,
        seed=int(seed),
    )
    train_ds, val_ds = full.split(float(val_fraction), seed=int(seed))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_edges = int(option_edges["edge_id"].max() + 1) if not option_edges.empty else 0
    model = GCBCMLP(
        obs_dim=full.obs_dim,
        action_dim=full.action_dim,
        hidden_dims=hidden_dims,
        use_remaining_h=use_remaining_h,
        remaining_h_scale=full.max_h,
        num_edges=max(1, num_edges),
        edge_embedding_dim=int(edge_embedding_dim),
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loader = _cycle(_make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=seed))
    log_interval = int(log_interval or max(1, min(1000, int(num_steps) // 10 if int(num_steps) > 0 else 1)))

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    source_rows_all: list[pd.DataFrame] = []
    last_edge_metrics = pd.DataFrame()
    last_source_metrics = pd.DataFrame()
    for step in range(1, int(num_steps) + 1):
        model.train()
        batch = _batch_to_device(next(loader), dev)
        model_edge_ids = batch["edge_id"].long().clamp(min=0, max=max(0, num_edges - 1))
        pred = model(batch["obs"], batch["goal"], batch["remaining_h"], model_edge_ids)
        unweighted_loss = action_mse(pred, batch["action"])
        loss = _weighted_action_mse(pred, batch["action"], batch["sample_weight"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_interval == 0 or step == int(num_steps):
            train_rows.append(
                {
                    "step": int(step),
                    "train_action_mse": float(unweighted_loss.detach().cpu().item()),
                    "train_weighted_action_mse": float(loss.detach().cpu().item()),
                }
            )
            val_mse, last_source_metrics, last_edge_metrics = evaluate_mixed_policy_mse(
                model,
                val_ds,
                batch_size=batch_size,
                max_examples=val_examples,
                device=dev,
            )
            source_summary = summarize_source_metrics(last_source_metrics)
            edge_summary = summarize_edge_groups(last_edge_metrics.merge(planner_weights, on="edge_id", how="left"), option_edges)
            val_rows.append({"step": int(step), "val_action_mse": float(val_mse), **source_summary, **edge_summary})
            step_sources = last_source_metrics.copy()
            if not step_sources.empty:
                step_sources.insert(0, "step", int(step))
                source_rows_all.append(step_sources)
            print(
                f"[phase5n] step={step} train_action_mse={float(unweighted_loss.detach().cpu().item()):.6g} "
                f"train_weighted_action_mse={float(loss.detach().cpu().item()):.6g} val_action_mse={val_mse:.6g}"
            )

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    source_df = pd.concat(source_rows_all, ignore_index=True) if source_rows_all else pd.DataFrame()
    edge_val = last_edge_metrics.merge(option_edges, on="edge_id", how="left") if not last_edge_metrics.empty else last_edge_metrics
    edge_val = edge_val.merge(planner_weights, on="edge_id", how="left") if not edge_val.empty else edge_val
    train_df.to_csv(out / "train_metrics.csv", index=False)
    val_df.to_csv(out / "val_metrics.csv", index=False)
    source_df.to_csv(out / "source_val_metrics.csv", index=False)
    edge_val.to_csv(out / "edge_val_metrics.csv", index=False)

    resolved = {
        **(config or {}),
        "output_dir": str(out),
        "batch_size": int(batch_size),
        "num_steps": int(num_steps),
        "lr": float(lr),
        "hidden_dims": hidden_dims,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "val_examples": int(val_examples),
        "train_examples": int(train_len),
        "obs_dim": int(full.obs_dim),
        "action_dim": int(full.action_dim),
        "num_edges": int(num_edges),
        "device": str(dev),
        "source_probabilities": full.source_probabilities,
        "source_loss_weights": full.source_loss_weights,
        "num_planner_queries": int(num_planner_queries),
        "base_loss_weight_mode": str(base_loss_weight_mode),
        "base_loss_weight_strength": float(base_loss_weight_strength),
        "planner_usage_strength": float(planner_usage_strength),
        "planner_first_edge_strength": float(planner_first_edge_strength),
        "min_loss_weight": float(min_loss_weight),
        "max_loss_weight": float(max_loss_weight),
        "note": "Offline supervised planner-aware GCBC metrics are not rollout success.",
    }
    try:
        import yaml

        with (out / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(_json_safe(resolved), f, sort_keys=True)
    except Exception:
        write_json(out / "config_resolved.json", resolved)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "obs_dim": full.obs_dim,
                "action_dim": full.action_dim,
                "hidden_dims": hidden_dims,
                "use_remaining_h": use_remaining_h,
                "remaining_h_scale": full.max_h,
                "num_edges": max(1, num_edges),
                "edge_embedding_dim": int(edge_embedding_dim),
            },
            "config": resolved,
        },
        out / "model.pt",
    )
    final_metrics = val_rows[-1] if val_rows else {}
    summary = {
        "phase": "Phase 5N",
        "title": "BARS Planner-Aware Low-Level GCBC",
        "final": final_metrics,
        "num_train_segments": int(train_ds.segment_indices.size),
        "num_val_segments": int(val_ds.segment_indices.size),
        "num_train_episodes": int(train_ds.episode_indices.size),
        "num_val_episodes": int(val_ds.episode_indices.size),
        "num_planner_reachable_queries": int(path_rows.get("reachable", pd.Series(dtype=bool)).astype(bool).sum()) if not path_rows.empty else 0,
        "mean_planner_loss_weight": float(planner_weights["loss_weight"].mean()) if not planner_weights.empty else np.nan,
        "note": "Offline supervised planner-aware GCBC metrics are not rollout success.",
    }
    write_json(out / "phase5n_train_summary.json", summary)
    plot_training_curves(train_df, val_df, out)
    return {
        "model": model,
        "train_metrics": train_df,
        "val_metrics": val_df,
        "source_val_metrics": source_df,
        "edge_val_metrics": edge_val,
        "planner_edge_weights": planner_weights,
        "planner_query_paths": path_rows,
        "output_dir": out,
    }


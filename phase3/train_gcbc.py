from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phase3.edge_bc_dataset import EdgeBCDataset, build_edge_bc_examples
from phase3.models import GCBCMLP, action_mse
from phase3.plotting import plot_training_curves

LOSS_WEIGHT_MODES = {"none", "support", "bottleneck", "support_bottleneck"}


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
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _make_loader(dataset: EdgeBCDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate_policy_mse(
    model: GCBCMLP,
    dataset: EdgeBCDataset,
    batch_size: int,
    max_examples: int,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    if len(dataset) == 0 or max_examples <= 0:
        return 0.0, pd.DataFrame(columns=["edge_id", "val_action_mse", "num_examples"])
    eval_ds = dataset.with_segment_indices(
        dataset.segment_indices,
        max_examples=min(int(max_examples), max(1, int(max_examples))),
        sampling_mode="uniform_transition",
        seed=dataset.seed + 991,
    )
    loader = _make_loader(eval_ds, batch_size=batch_size, shuffle=False, seed=dataset.seed + 13)
    model.eval()
    total = 0.0
    count = 0
    edge_sum: dict[int, float] = {}
    edge_count: dict[int, int] = {}
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            pred = model(batch["obs"], batch["goal"], batch["remaining_h"], batch["edge_id"])
            per_sample = torch.mean((pred - batch["action"].float()) ** 2, dim=1)
            values = per_sample.detach().cpu().numpy()
            edge_ids = batch["edge_id"].detach().cpu().numpy()
            total += float(values.sum())
            count += int(values.shape[0])
            for edge_id, value in zip(edge_ids, values):
                edge_sum[int(edge_id)] = edge_sum.get(int(edge_id), 0.0) + float(value)
                edge_count[int(edge_id)] = edge_count.get(int(edge_id), 0) + 1
            if count >= int(max_examples):
                break
    rows = [
        {
            "edge_id": int(edge_id),
            "val_action_mse": float(edge_sum[edge_id] / max(1, edge_count[edge_id])),
            "num_examples": int(edge_count[edge_id]),
        }
        for edge_id in sorted(edge_sum)
    ]
    return float(total / max(1, count)), pd.DataFrame(rows)


def summarize_edge_val_metrics(edge_metrics: pd.DataFrame, option_edges: pd.DataFrame) -> dict[str, float]:
    if edge_metrics.empty:
        return {
            "bottleneck_edge_val_mse": 0.0,
            "non_bottleneck_edge_val_mse": 0.0,
            "high_support_edge_val_mse": 0.0,
            "low_support_edge_val_mse": 0.0,
            "short_horizon_edge_val_mse": 0.0,
            "long_horizon_edge_val_mse": 0.0,
        }
    merged = edge_metrics.merge(option_edges, on="edge_id", how="left", suffixes=("", "_edge"))
    bottleneck_thr = float(merged["edge_bottleneck_score"].median()) if "edge_bottleneck_score" in merged else 0.0
    support_col = "num_unique_starts" if "num_unique_starts" in merged else "num_segments"
    support_thr = float(merged[support_col].median()) if support_col in merged else 0.0
    horizon_col = "median_h" if "median_h" in merged else None
    horizon_thr = float(merged[horizon_col].median()) if horizon_col is not None else 0.0

    def _mean(mask: pd.Series) -> float:
        sub = merged[mask]
        return float(sub["val_action_mse"].mean()) if not sub.empty else 0.0

    return {
        "bottleneck_edge_val_mse": _mean(merged["edge_bottleneck_score"] >= bottleneck_thr)
        if "edge_bottleneck_score" in merged
        else 0.0,
        "non_bottleneck_edge_val_mse": _mean(merged["edge_bottleneck_score"] < bottleneck_thr)
        if "edge_bottleneck_score" in merged
        else 0.0,
        "high_support_edge_val_mse": _mean(merged[support_col] >= support_thr) if support_col in merged else 0.0,
        "low_support_edge_val_mse": _mean(merged[support_col] < support_thr) if support_col in merged else 0.0,
        "short_horizon_edge_val_mse": _mean(merged[horizon_col] <= horizon_thr)
        if horizon_col is not None
        else 0.0,
        "long_horizon_edge_val_mse": _mean(merged[horizon_col] > horizon_thr)
        if horizon_col is not None
        else 0.0,
    }


def edge_loss_weight_values(
    option_edges: pd.DataFrame,
    mode: str = "none",
    strength: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 3.0,
) -> pd.DataFrame:
    """Build per-edge supervised loss weights from Phase 2 metadata."""

    mode = str(mode or "none")
    if mode not in LOSS_WEIGHT_MODES:
        raise ValueError(f"loss_weight_mode must be one of {sorted(LOSS_WEIGHT_MODES)}")
    if option_edges.empty:
        return pd.DataFrame(columns=["edge_id", "loss_weight"])
    out = option_edges[["edge_id"]].copy()
    if mode == "none":
        out["loss_weight"] = 1.0
        return out

    def _support_component() -> np.ndarray:
        support_col = "num_unique_starts" if "num_unique_starts" in option_edges.columns else "num_segments"
        support = pd.to_numeric(option_edges.get(support_col, 1.0), errors="coerce").fillna(1.0).clip(lower=1.0)
        return 1.0 / np.sqrt(support.to_numpy(dtype=np.float64))

    def _bottleneck_component() -> np.ndarray:
        if "edge_bottleneck_score" not in option_edges.columns:
            return np.ones(option_edges.shape[0], dtype=np.float64)
        values = pd.to_numeric(option_edges["edge_bottleneck_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
        arr = values.to_numpy(dtype=np.float64)
        if arr.size == 0:
            return np.ones(option_edges.shape[0], dtype=np.float64)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            return np.ones_like(arr, dtype=np.float64)
        return 0.5 + (arr - lo) / (hi - lo)

    raw = np.ones(option_edges.shape[0], dtype=np.float64)
    if mode in {"support", "support_bottleneck"}:
        raw *= _support_component()
    if mode in {"bottleneck", "support_bottleneck"}:
        raw *= _bottleneck_component()
    if raw.size == 0 or not np.isfinite(raw).all() or float(raw.mean()) <= 0.0:
        normalized = np.ones(option_edges.shape[0], dtype=np.float64)
    else:
        normalized = raw / float(raw.mean())
    weights = 1.0 + float(strength) * (normalized - 1.0)
    lo = min(float(min_weight), float(max_weight))
    hi = max(float(min_weight), float(max_weight))
    out["loss_weight"] = np.clip(weights, lo, hi)
    return out


def make_edge_loss_weight_tensor(
    option_edges: pd.DataFrame,
    num_edges: int,
    device: torch.device,
    mode: str = "none",
    strength: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 3.0,
) -> torch.Tensor:
    values = edge_loss_weight_values(
        option_edges,
        mode=mode,
        strength=strength,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    weights = np.ones(max(1, int(num_edges)), dtype=np.float32)
    for row in values.itertuples(index=False):
        edge_id = int(row.edge_id)
        if 0 <= edge_id < weights.shape[0]:
            weights[edge_id] = float(row.loss_weight)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def weighted_action_mse(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    per_sample = torch.mean((pred - target.float()) ** 2, dim=1)
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).clamp(min=1e-8)
    return torch.sum(per_sample * weights) / torch.sum(weights)


def train_gcbc(
    dataset: dict[str, Any],
    option_edges_csv: str | Path | pd.DataFrame,
    edge_segments_npz: str | Path | dict[str, np.ndarray],
    output_dir: str | Path,
    sampling_mode: str = "uniform_edge",
    batch_size: int = 1024,
    num_steps: int = 100000,
    lr: float = 3e-4,
    hidden_dims: list[int] | str | None = None,
    seed: int = 0,
    val_fraction: float = 0.05,
    val_examples: int = 8192,
    log_interval: int | None = None,
    device: str | None = None,
    use_remaining_h: bool = True,
    edge_embedding_dim: int = 0,
    loss_weight_mode: str = "none",
    loss_weight_strength: float = 1.0,
    loss_weight_min: float = 0.25,
    loss_weight_max: float = 3.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = build_edge_bc_examples(
        dataset,
        option_edges_csv,
        edge_segments_npz,
        max_examples=None,
        sampling_mode=sampling_mode,  # type: ignore[arg-type]
        seed=seed,
    )
    train_ds, val_ds = full.split_by_segments(float(val_fraction), seed=seed)
    option_edges = full.option_edges.copy()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_edges = int(option_edges["edge_id"].max() + 1) if not option_edges.empty else 0
    model = GCBCMLP(
        obs_dim=full.obs_dim,
        action_dim=full.action_dim,
        hidden_dims=hidden_dims,
        use_remaining_h=use_remaining_h,
        remaining_h_scale=full.max_h,
        num_edges=num_edges,
        edge_embedding_dim=int(edge_embedding_dim),
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loader = _cycle(_make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=seed))
    log_interval = int(log_interval or max(1, min(1000, int(num_steps) // 10 if int(num_steps) > 0 else 1)))
    edge_loss_weights = make_edge_loss_weight_tensor(
        option_edges,
        num_edges=max(1, num_edges),
        device=dev,
        mode=loss_weight_mode,
        strength=float(loss_weight_strength),
        min_weight=float(loss_weight_min),
        max_weight=float(loss_weight_max),
    )

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    last_edge_metrics = pd.DataFrame()
    for step in range(1, int(num_steps) + 1):
        model.train()
        batch = _batch_to_device(next(loader), dev)
        pred = model(batch["obs"], batch["goal"], batch["remaining_h"], batch["edge_id"])
        unweighted_loss = action_mse(pred, batch["action"])
        edge_ids = batch["edge_id"].long().clamp(min=0, max=edge_loss_weights.shape[0] - 1)
        loss = weighted_action_mse(pred, batch["action"], edge_loss_weights[edge_ids])
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
            val_mse, last_edge_metrics = evaluate_policy_mse(
                model,
                val_ds,
                batch_size=batch_size,
                max_examples=val_examples,
                device=dev,
            )
            edge_summary = summarize_edge_val_metrics(last_edge_metrics, option_edges)
            val_rows.append({"step": int(step), "val_action_mse": float(val_mse), **edge_summary})
            print(
                f"[phase3] step={step} train_action_mse={float(unweighted_loss.detach().cpu().item()):.6g} "
                f"train_weighted_action_mse={float(loss.detach().cpu().item()):.6g} "
                f"val_action_mse={val_mse:.6g}"
            )

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    edge_val = last_edge_metrics.merge(option_edges, on="edge_id", how="left") if not last_edge_metrics.empty else last_edge_metrics
    train_df.to_csv(out / "train_metrics.csv", index=False)
    val_df.to_csv(out / "val_metrics.csv", index=False)
    edge_val.to_csv(out / "edge_val_metrics.csv", index=False)
    resolved = {
        **(config or {}),
        "output_dir": str(out),
        "sampling_mode": sampling_mode,
        "batch_size": int(batch_size),
        "num_steps": int(num_steps),
        "lr": float(lr),
        "hidden_dims": hidden_dims,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "val_examples": int(val_examples),
        "obs_dim": int(full.obs_dim),
        "action_dim": int(full.action_dim),
        "num_edges": int(num_edges),
        "device": str(dev),
        "loss_weight_mode": str(loss_weight_mode),
        "loss_weight_strength": float(loss_weight_strength),
        "loss_weight_min": float(loss_weight_min),
        "loss_weight_max": float(loss_weight_max),
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
                "num_edges": num_edges,
                "edge_embedding_dim": int(edge_embedding_dim),
            },
            "config": resolved,
        },
        out / "model.pt",
    )
    final_metrics = val_rows[-1] if val_rows else {}
    write_json(out / "train_summary.json", {"final": final_metrics, "num_train_segments": int(train_ds.segment_indices.size)})
    plot_training_curves(train_df, val_df, out)
    return {
        "model": model,
        "train_metrics": train_df,
        "val_metrics": val_df,
        "edge_val_metrics": edge_val,
        "output_dir": out,
    }

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReachabilityMLP(nn.Module):
    def __init__(self, phi_dim: int, hidden: tuple[int, ...] = (256, 256), dropout: float = 0.0):
        super().__init__()
        in_dim = phi_dim * 3 + 1
        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden:
            layers.extend([nn.Linear(cur, h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u_phi: torch.Tensor, v_phi: torch.Tensor) -> torch.Tensor:
        delta = v_phi - u_phi
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        x = torch.cat([u_phi, v_phi, delta, dist], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def prob(self, u_phi: torch.Tensor, v_phi: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(u_phi, v_phi))


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(np.int32)
    out: dict[str, float] = {}
    eps = 1e-7
    out["bce"] = float(-(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))).mean())
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        out["auprc"] = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(p))
        pos = y == 1
        neg = y == 0
        if pos.any() and neg.any():
            out["auroc"] = float((ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * neg.sum()))
        else:
            out["auroc"] = float("nan")
        desc = np.argsort(-p)
        yy = y[desc]
        tp = np.cumsum(yy == 1)
        fp = np.cumsum(yy == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall_step = (yy == 1) / max(int(pos.sum()), 1)
        out["auprc"] = float(np.sum(precision * recall_step))
    for k in (50, 100, 500, 1000):
        kk = min(k, len(p))
        if kk:
            idx = np.argsort(-p)[:kk]
            out[f"precision_at_{k}"] = float(y[idx].mean())
    return out


def _reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            rows.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": int(mask.sum()),
                    "p_mean": float(p[mask].mean()),
                    "y_mean": float(y[mask].mean()),
                }
            )
    return rows


def _tensorize(df: pd.DataFrame, phis: np.ndarray, split: str, device: str):
    mask = (df["split"].astype(str) == split) & (df["y"] >= 0)
    part = df.loc[mask]
    u = torch.as_tensor(phis[part["u_idx"].to_numpy(np.int64)], dtype=torch.float32, device=device)
    v = torch.as_tensor(phis[part["v_idx"].to_numpy(np.int64)], dtype=torch.float32, device=device)
    y = torch.as_tensor(part["y"].to_numpy(np.float32), dtype=torch.float32, device=device)
    w = torch.as_tensor(part["weight"].to_numpy(np.float32), dtype=torch.float32, device=device)
    return u, v, y, w


def train_reachability(
    pairs: pd.DataFrame,
    phis: np.ndarray,
    out_dir: str | os.PathLike[str],
    quick: bool = True,
    device: str = "cpu",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: float = 1e-3,
    hidden: tuple[int, ...] = (256, 256),
    mixed_precision: bool = False,
) -> tuple[ReachabilityMLP, dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    epochs = int(epochs or (5 if quick else 12))
    batch_size = int(batch_size or (8192 if quick else 16384))
    phis = np.asarray(phis, dtype=np.float32)
    model = ReachabilityMLP(phis.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    u, v, y, w = _tensorize(pairs, phis, "train", device)
    if len(y) == 0:
        raise ValueError("No labeled training reachability pairs")
    loader = DataLoader(TensorDataset(u, v, y, w), batch_size=batch_size, shuffle=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device == "cuda"))
    writer = None
    if os.environ.get("BARS_USE_TENSORBOARD", "").lower() in {"1", "true", "yes"}:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_root = Path(os.environ.get("TENSORBOARD_LOGDIR", "runs_stage22_tensorboard"))
            writer = SummaryWriter(log_dir=str(tb_root / "reachability" / out.name))
        except Exception as exc:
            print(f"[reachability] TensorBoard unavailable ({exc!r}); continuing with JSON metrics.")
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bu, bv, by, bw in loader:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(mixed_precision and device == "cuda")):
                logits = model(bu, bv)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, by, weight=bw)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        train_bce = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_bce": train_bce})
        if writer is not None:
            writer.add_scalar("train/bce", train_bce, epoch)
    metrics = evaluate_reachability(model, pairs, phis, device=device)
    if writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f"metrics/{key}", float(value), epochs)
        writer.flush()
        writer.close()
    metrics["history"] = history
    metrics["train_wall_sec"] = time.time() - start
    torch.save({"model": model.state_dict(), "phi_dim": phis.shape[1], "hidden": hidden}, out / "model.pt")
    calibration = {
        "metrics": metrics,
        "reliability": metrics.pop("reliability_bins", []),
        "quick": bool(quick),
        "epochs": epochs,
        "batch_size": batch_size,
    }
    with open(out / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2, sort_keys=True)
    with open(out / "reachability_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return model, metrics


@torch.no_grad()
def evaluate_reachability(model: ReachabilityMLP, pairs: pd.DataFrame, phis: np.ndarray, device: str = "cpu") -> dict[str, Any]:
    device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out: dict[str, Any] = {}
    for split in ("train", "val"):
        mask = (pairs["split"].astype(str) == split) & (pairs["y"] >= 0)
        part = pairs.loc[mask]
        if len(part) == 0:
            continue
        probs = []
        for st in range(0, len(part), 65536):
            chunk = part.iloc[st : st + 65536]
            u = torch.as_tensor(phis[chunk["u_idx"].to_numpy(np.int64)], dtype=torch.float32, device=device)
            v = torch.as_tensor(phis[chunk["v_idx"].to_numpy(np.int64)], dtype=torch.float32, device=device)
            probs.append(model.prob(u, v).cpu().numpy())
        p = np.concatenate(probs)
        y = part["y"].to_numpy(np.float32)
        for k, v in _binary_metrics(y, p).items():
            out[f"{split}_{k}"] = v
        if split == "val":
            out["reliability_bins"] = _reliability(y, p)
    return out


def load_reachability_model(path: str | os.PathLike[str], device: str = "cpu") -> ReachabilityMLP:
    ckpt = torch.load(path, map_location=device)
    model = ReachabilityMLP(int(ckpt["phi_dim"]), hidden=tuple(ckpt.get("hidden", (256, 256))))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def score_edges(
    model: ReachabilityMLP,
    node_phis: np.ndarray,
    edge_table: pd.DataFrame,
    device: str = "cpu",
    batch_size: int = 65536,
) -> pd.DataFrame:
    device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    u_idx = edge_table["u"].to_numpy(np.int64)
    v_idx = edge_table["v"].to_numpy(np.int64)
    probs = []
    for st in range(0, len(edge_table), batch_size):
        sl = slice(st, st + batch_size)
        u = torch.as_tensor(node_phis[u_idx[sl]], dtype=torch.float32, device=device)
        v = torch.as_tensor(node_phis[v_idx[sl]], dtype=torch.float32, device=device)
        probs.append(model.prob(u, v).cpu().numpy())
    p = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
    out = edge_table.copy()
    out["p_exec"] = p
    out["r_exec"] = -np.log(np.clip(p, 1e-6, 1.0))
    return out


def edge_score_diagnostics(edge_scores: pd.DataFrame) -> dict[str, Any]:
    p = edge_scores["p_exec"].to_numpy(np.float32) if "p_exec" in edge_scores else np.empty(0)
    out: dict[str, Any] = {"num_edges": int(len(edge_scores))}
    if len(p):
        for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
            out[f"p_exec_q{int(q * 100):02d}"] = float(np.quantile(p, q))
        out["p_exec_mean"] = float(np.mean(p))
    if "local_support" in edge_scores:
        local = edge_scores["local_support"].to_numpy(np.float32)
        out["local_support_rate"] = float(local.mean()) if len(local) else 0.0
    if "same_traj_support" in edge_scores:
        support = edge_scores["same_traj_support"].to_numpy(np.float32)
        out["same_traj_support_mean"] = float(support.mean()) if len(support) else 0.0
    if "r_exec" in edge_scores:
        risk = edge_scores["r_exec"].to_numpy(np.float32)
        for q in (0.5, 0.75, 0.9, 0.95, 0.99):
            out[f"r_exec_q{int(q * 100):02d}"] = float(np.quantile(risk, q)) if len(risk) else 0.0
    return out

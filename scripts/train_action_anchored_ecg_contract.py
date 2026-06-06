#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_src" / "GAS"))

from cage.contract_model import (  # noqa: E402
    LinearContractModel,
    auprc,
    auroc,
    build_contract_feature_vector,
    sigmoid,
    train_logistic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an action-anchored ECG contract model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_examples", type=int, default=500000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_examples(Path(args.dataset), max_examples=int(args.max_examples))
    metrics: dict[str, Any] = {"dataset": str(args.dataset), "num_examples": len(rows)}
    if not rows:
        metrics["status"] = "CONTRACT_MODEL_BLOCKED_EMPTY_DATASET"
        write_outputs(out_dir, None, metrics)
        return 2
    x = np.stack([build_contract_feature_vector(row) for row in rows]).astype(np.float32)
    labels = {
        "contract_positive": np.asarray([bool(row.get("label_positive_contract")) for row in rows], dtype=np.float32),
        "negative_progress": np.asarray([bool(row.get("label_negative_contract")) for row in rows], dtype=np.float32),
        "final_goal": np.asarray([bool(row.get("label_final_goal")) for row in rows], dtype=np.float32),
        "hit": np.asarray([bool(row.get("label_positive_contract")) for row in rows], dtype=np.float32),
    }
    d_phi = np.asarray([float(row.get("d_phi", 0.0) or 0.0) for row in rows], dtype=np.float32)
    target_modes = [str(row.get("target_mode", "unknown")) for row in rows]
    horizons = np.asarray([int(row.get("horizon", -1) or -1) for row in rows], dtype=np.int32)
    split = split_indices(len(rows), seed=int(args.seed), val_ratio=float(args.val_ratio), test_ratio=float(args.test_ratio))
    x_train = x[split["train"]]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    xz = (x - mean) / np.maximum(std, 1e-6)
    weights: dict[str, list[float]] = {}
    metrics["splits"] = {key: int(len(value)) for key, value in split.items()}
    metrics["target_mode_counts"] = count_values(target_modes)
    metrics["horizon_counts"] = count_values([int(h) for h in horizons])
    for target, y in labels.items():
        target_metrics, w = train_eval_target(
            xz=xz,
            y=y,
            d_phi=d_phi,
            split=split,
            epochs=max(20, int(args.epochs)),
            target=target,
        )
        metrics[target] = target_metrics
        if w is not None:
            weights[target] = w.tolist()
    if "contract_positive" not in weights or "negative_progress" not in weights:
        metrics["status"] = "CONTRACT_MODEL_UNDERPOWERED_SINGLE_CLASS"
        model = None
    else:
        model = LinearContractModel(mean=mean.tolist(), std=std.tolist(), weights=weights)
        pos = metrics["contract_positive"]
        neg = metrics["negative_progress"]
        pos_ok = compare_metric(pos, "auroc_test", "dphi_auroc_test", higher=True)
        neg_ok = compare_metric(neg, "auroc_test", "dphi_auroc_test", higher=True)
        metrics["gate"] = {
            "positive_contract_beats_dphi": pos_ok,
            "negative_contract_beats_dphi": neg_ok,
        }
        metrics["status"] = "CONTRACT_MODEL_READY" if pos_ok else "CONTRACT_MODEL_FAILS_DPHI_BASELINE"
    metrics["calibration"] = calibration_table(xz, labels["contract_positive"], weights.get("contract_positive"), split["test"])
    metrics["selective_prediction_curve"] = selective_curve(xz, labels["contract_positive"], weights.get("contract_positive"), split["test"])
    metrics["group_metrics"] = group_metrics(xz, labels["contract_positive"], weights.get("contract_positive"), split["test"], target_modes, horizons)
    write_outputs(out_dir, model, metrics)
    print(json.dumps({"status": metrics["status"], "out_dir": str(out_dir)}, sort_keys=True))
    return 0 if metrics["status"] == "CONTRACT_MODEL_READY" else 2


def load_examples(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("action_available") is not True:
                continue
            rows.append(
                {
                    "phi_s": rec.get("phi_s"),
                    "phi_g": rec.get("phi_g"),
                    "d_phi": rec.get("d_phi"),
                    "target_mode": rec.get("target_mode"),
                    "env_name": rec.get("env_name"),
                    "label_positive_contract": rec.get("label_positive_contract"),
                    "label_negative_contract": rec.get("label_negative_contract"),
                    "label_final_goal": rec.get("label_final_goal"),
                    "final_phase": rec.get("label_final_goal"),
                    "horizon": rec.get("horizon"),
                }
            )
    return rows


def split_indices(n: int, *, seed: int, val_ratio: float, test_ratio: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return {"train": train, "val": val, "test": test}


def train_eval_target(
    *,
    xz: np.ndarray,
    y: np.ndarray,
    d_phi: np.ndarray,
    split: dict[str, np.ndarray],
    epochs: int,
    target: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    out: dict[str, Any] = {"positive_rate": float(y.mean()) if len(y) else None}
    train_idx = split["train"]
    if len(np.unique(y[train_idx])) < 2:
        out["status"] = "single_class_train"
        return out, None
    w = train_logistic(xz[train_idx], y[train_idx], epochs=epochs, lr=0.05)
    if w is None:
        out["status"] = "single_class_train"
        return out, None
    scores = sigmoid(np.concatenate([xz, np.ones((len(xz), 1), dtype=np.float32)], axis=1) @ w)
    dphi_score = -d_phi if target != "negative_progress" else d_phi
    for name, idx in split.items():
        out[f"auroc_{name}"] = auroc(y[idx], scores[idx])
        out[f"auprc_{name}"] = auprc(y[idx], scores[idx])
        out[f"brier_{name}"] = brier(y[idx], scores[idx])
        out[f"dphi_auroc_{name}"] = auroc(y[idx], dphi_score[idx])
        out[f"dphi_auprc_{name}"] = auprc(y[idx], dphi_score[idx])
    out["status"] = "trained"
    return out, w


def brier(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def compare_metric(row: dict[str, Any], model_key: str, baseline_key: str, *, higher: bool) -> bool:
    model = row.get(model_key)
    base = row.get(baseline_key)
    if model is None or base is None:
        return False
    return bool(model > base) if higher else bool(model < base)


def calibration_table(xz: np.ndarray, y: np.ndarray, weights: list[float] | None, idx: np.ndarray) -> list[dict[str, Any]]:
    if weights is None or len(idx) == 0:
        return []
    scores = predict_with_weights(xz, weights)[idx]
    labels = y[idx]
    rows: list[dict[str, Any]] = []
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        rows.append({"bin_start": float(lo), "bin_end": float(hi), "count": int(mask.sum()), "mean_pred": mean_or_none(scores[mask]), "empirical_rate": mean_or_none(labels[mask])})
    return rows


def selective_curve(xz: np.ndarray, y: np.ndarray, weights: list[float] | None, idx: np.ndarray) -> list[dict[str, Any]]:
    if weights is None or len(idx) == 0:
        return []
    scores = predict_with_weights(xz, weights)[idx]
    labels = y[idx]
    rows: list[dict[str, Any]] = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = scores >= threshold
        rows.append({"threshold": threshold, "coverage": safe_rate(int(mask.sum()), len(mask)), "empirical_positive_rate": mean_or_none(labels[mask])})
    return rows


def group_metrics(xz: np.ndarray, y: np.ndarray, weights: list[float] | None, idx: np.ndarray, modes: list[str], horizons: np.ndarray) -> list[dict[str, Any]]:
    if weights is None or len(idx) == 0:
        return []
    scores = predict_with_weights(xz, weights)
    rows: list[dict[str, Any]] = []
    for mode in sorted(set(modes)):
        mask = np.asarray([m == mode for m in modes]) & np.isin(np.arange(len(y)), idx)
        if mask.any():
            rows.append({"group": f"target_mode={mode}", "count": int(mask.sum()), "positive_rate": mean_or_none(y[mask]), "mean_score": mean_or_none(scores[mask])})
    for horizon in sorted(set(int(h) for h in horizons if int(h) >= 0)):
        mask = (horizons == horizon) & np.isin(np.arange(len(y)), idx)
        if mask.any():
            rows.append({"group": f"horizon={horizon}", "count": int(mask.sum()), "positive_rate": mean_or_none(y[mask]), "mean_score": mean_or_none(scores[mask])})
    return rows


def predict_with_weights(xz: np.ndarray, weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    xb = np.concatenate([xz, np.ones((len(xz), 1), dtype=np.float32)], axis=1)
    return sigmoid(xb @ w)


def mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def safe_rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def count_values(values: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def write_outputs(out_dir: Path, model: LinearContractModel | None, metrics: dict[str, Any]) -> None:
    metrics_path = out_dir / "metrics.json"
    report_path = REPO_ROOT / "reports" / "stage38_action_anchored_contract_model.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if model is not None:
        payload = model.to_dict()
        payload["stage"] = "stage38_action_anchored_ecg_contract"
        for path in [out_dir / "model.json", out_dir / "model.pt"]:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage38 Action-Anchored ECG Contract Model",
        "",
        f"- status: `{metrics.get('status')}`",
        f"- examples: {metrics.get('num_examples')}",
        f"- model: `{out_dir / 'model.pt'}`" if model is not None else "- model: `NA`",
        f"- metrics: `{metrics_path}`",
        "",
        "## Gate",
        "",
        f"- positive_contract_beats_dphi: `{metrics.get('gate', {}).get('positive_contract_beats_dphi')}`",
        f"- negative_contract_beats_dphi: `{metrics.get('gate', {}).get('negative_contract_beats_dphi')}`",
        "",
        "## Test Metrics",
        "",
    ]
    for target in ["contract_positive", "negative_progress", "final_goal"]:
        row = metrics.get(target, {})
        lines.append(f"- {target}: status={row.get('status')} auroc_test={row.get('auroc_test')} dphi_auroc_test={row.get('dphi_auroc_test')} brier_test={row.get('brier_test')}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

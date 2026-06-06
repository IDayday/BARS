#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight ECG goal-conditioned policy adapter from action-anchored positives.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contract_model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_examples", type=int, default=200000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_positive_examples(Path(args.dataset), set(args.envs), max_examples=int(args.max_train_examples))
    metrics: dict[str, Any] = {
        "dataset": str(args.dataset),
        "contract_model": str(args.contract_model),
        "num_positive_examples": len(rows),
        "envs": list(args.envs),
    }
    if len(rows) < 2:
        metrics["status"] = "POLICY_ADAPTER_BLOCKED_NO_POSITIVE_ACTIONS"
        write_outputs(out_dir, None, metrics)
        return 2
    x, y = make_xy(rows)
    split = split_indices(len(rows), seed=int(args.seed))
    model = train_ridge_adapter(x[split["train"]], y[split["train"]], ridge=1e-3)
    pred_train = predict(model, x[split["train"]])
    pred_val = predict(model, x[split["val"]])
    pred_test = predict(model, x[split["test"]])
    mean_action = y[split["train"]].mean(axis=0, keepdims=True)
    zero_action = np.zeros_like(mean_action)
    metrics["splits"] = {key: int(len(value)) for key, value in split.items()}
    metrics["input_dim"] = int(x.shape[1])
    metrics["action_dim"] = int(y.shape[1])
    metrics["underpowered"] = bool(len(rows) < 1000)
    metrics["train_mse"] = mse(pred_train, y[split["train"]])
    metrics["val_mse"] = mse(pred_val, y[split["val"]])
    metrics["test_mse"] = mse(pred_test, y[split["test"]])
    metrics["zero_action_val_mse"] = mse(np.repeat(zero_action, len(split["val"]), axis=0), y[split["val"]])
    metrics["mean_action_val_mse"] = mse(np.repeat(mean_action, len(split["val"]), axis=0), y[split["val"]])
    metrics["zero_action_test_mse"] = mse(np.repeat(zero_action, len(split["test"]), axis=0), y[split["test"]])
    metrics["mean_action_test_mse"] = mse(np.repeat(mean_action, len(split["test"]), axis=0), y[split["test"]])
    metrics["gate"] = {
        "beats_mean_action_baseline": bool(metrics["val_mse"] < metrics["mean_action_val_mse"]),
        "beats_zero_action_baseline": bool(metrics["val_mse"] < metrics["zero_action_val_mse"]),
    }
    metrics["status"] = "POLICY_ADAPTER_READY" if metrics["gate"]["beats_mean_action_baseline"] else "POLICY_ADAPTER_FAILS_MEAN_BASELINE"
    write_outputs(out_dir, model, metrics)
    print(json.dumps({"status": metrics["status"], "out_dir": str(out_dir)}, sort_keys=True))
    return 0 if metrics["status"] == "POLICY_ADAPTER_READY" else 2


def load_positive_examples(path: Path, envs: set[str], *, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("env_name") not in envs:
                continue
            if not rec.get("action_available") or not rec.get("trainable_for_bc"):
                continue
            if not rec.get("label_positive_contract"):
                continue
            rows.append(rec)
    return rows


def make_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for rec in rows:
        obs = np.asarray(rec["observation"], dtype=np.float32)
        phi_s = np.asarray(rec["phi_s"], dtype=np.float32)
        phi_g = np.asarray(rec["phi_g"], dtype=np.float32)
        x_rows.append(np.concatenate([obs, phi_s, phi_g, phi_g - phi_s, np.abs(phi_g - phi_s)], axis=0))
        y_rows.append(np.asarray(rec["action"], dtype=np.float32))
    return np.stack(x_rows), np.stack(y_rows)


def split_indices(n: int, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * 0.1)))
    n_val = max(1, int(round(n * 0.1)))
    return {"test": idx[:n_test], "val": idx[n_test:n_test + n_val], "train": idx[n_test + n_val:]}


def train_ridge_adapter(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, Any]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    xz = (x - mean) / np.maximum(std, 1e-6)
    xb = np.concatenate([xz, np.ones((len(xz), 1), dtype=np.float32)], axis=1)
    reg = float(ridge) * np.eye(xb.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return {
        "model_type": "linear_ecg_policy_adapter",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights[:-1].tolist(),
        "bias": weights[-1].tolist(),
    }


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = np.asarray(model["bias"], dtype=np.float32)
    xz = (x - mean) / np.maximum(std, 1e-6)
    return xz @ weights + bias


def mse(pred: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean((pred - y) ** 2))


def write_outputs(out_dir: Path, model: dict[str, Any] | None, metrics: dict[str, Any]) -> None:
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if model is not None:
        payload = dict(model)
        payload["stage"] = "stage38_action_anchored_ecg_policy_adapter"
        for path in [out_dir / "model.json", out_dir / "model.pt"]:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = REPO_ROOT / "reports" / "stage38_ecg_policy_adapter_train.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage38 ECG Policy Adapter Training",
        "",
        f"- status: `{metrics.get('status')}`",
        f"- positive examples: {metrics.get('num_positive_examples')}",
        f"- underpowered: `{metrics.get('underpowered')}`",
        f"- val_mse: {metrics.get('val_mse')}",
        f"- mean_action_val_mse: {metrics.get('mean_action_val_mse')}",
        f"- zero_action_val_mse: {metrics.get('zero_action_val_mse')}",
        f"- model: `{out_dir / 'model.pt'}`" if model is not None else "- model: `NA`",
        f"- metrics: `{metrics_path}`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

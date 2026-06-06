#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GAS_ROOT = Path(__file__).resolve().parent / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.closed_loop_contract_dataset import ClosedLoopContractDataset  # noqa: E402
from cage.contract_model import auprc, auroc, save_model, train_contract_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal offline linear contract predictor on CLP contract labels.")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--train_path", default="")
    parser.add_argument("--val_path", default="")
    parser.add_argument("--test_path", default="")
    parser.add_argument("--out_model", default="results/cage_v02_contract/models/contract_model.json")
    parser.add_argument("--out_metrics", default="results/cage_v02_contract/models/eval_metrics.json")
    parser.add_argument("--out_report", default="results/cage_v02_contract/models/eval_report.md")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--min_examples", type=int, default=100)
    return parser.parse_args()


def load_dataset(path: str) -> ClosedLoopContractDataset:
    return ClosedLoopContractDataset.from_jsonl(path)


def brier(y, pred):
    import numpy as np

    if len(y) == 0:
        return None
    return float(np.mean((pred - y) ** 2))


def calibration_table(y, pred, bins: int = 5):
    import numpy as np

    rows = []
    if len(y) == 0:
        return rows
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred >= lo) & (pred <= hi if hi == 1.0 else pred < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": f"{lo:.2f}-{hi:.2f}",
                "count": int(np.sum(mask)),
                "mean_pred": float(np.mean(pred[mask])),
                "empirical_rate": float(np.mean(y[mask])),
            }
        )
    return rows


def selective_curve(y, pred):
    import numpy as np

    if len(y) == 0:
        return []
    order = np.argsort(-pred)
    rows = []
    for coverage in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = max(1, int(round(len(y) * coverage)))
        idx = order[:k]
        rows.append({"coverage": coverage, "empirical_success": float(np.mean(y[idx])), "num_examples": int(k)})
    return rows


def evaluate_model(model, dataset: ClosedLoopContractDataset, split_name: str) -> dict:
    import numpy as np

    x, labels = dataset.feature_matrix()
    metrics = {"num_examples": int(len(x))}
    if len(x) == 0 or model is None:
        metrics["status"] = "empty_or_no_model"
        return metrics
    d_phi = x[:, -1] if x.shape[1] else np.empty((len(x),), dtype=np.float32)
    for target in ["hit", "contract_positive", "negative_progress"]:
        y = labels.get(target)
        if y is None or len(y) != len(x):
            continue
        if len(np.unique(y)) < 2:
            metrics[f"{target}_status"] = "single_class"
            continue
        if target in model.weights:
            pred = model.predict_proba(x, target)
            metrics[f"{target}_auroc"] = auroc(y, pred)
            metrics[f"{target}_auprc"] = auprc(y, pred)
            metrics[f"{target}_brier"] = brier(y, pred)
            metrics[f"{target}_calibration"] = calibration_table(y, pred)
            metrics[f"{target}_selective_prediction"] = selective_curve(y, pred)
        baseline_score = d_phi if target == "negative_progress" else -d_phi
        metrics[f"{target}_d_phi_baseline_auroc"] = auroc(y, baseline_score)
    metrics["status"] = "ok"
    metrics["split"] = split_name
    return metrics


def write_report(path: str, metrics: dict) -> None:
    lines = [
        "# CAGE Contract Model Evaluation",
        "",
        f"- status: {metrics.get('status')}",
        f"- out_model: `{metrics.get('out_model')}`",
        "",
        "| split | examples | hit AUROC | contract+ AUROC | neg-progress AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        row = metrics.get(split, {})
        lines.append(
            f"| {split} | {row.get('num_examples', 0)} | {fmt(row.get('hit_auroc'))} | "
            f"{fmt(row.get('contract_positive_auroc'))} | {fmt(row.get('negative_progress_auroc'))} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    return "NA" if value is None else f"{float(value):.3f}"


def main() -> int:
    args = parse_args()
    train_path = args.train_path or args.dataset_path
    if not train_path:
        raise SystemExit("--train_path or --dataset_path is required")
    train_dataset = load_dataset(train_path)
    x, labels = train_dataset.feature_matrix()
    model, metrics = train_contract_model(x, labels, epochs=args.epochs, lr=args.lr)
    out_metrics = {"train_fit": metrics, "out_model": None, "status": metrics.get("status")}
    if model is not None:
        save_model(args.out_model, model)
        out_metrics["out_model"] = args.out_model
    split_paths = {"train": train_path, "val": args.val_path, "test": args.test_path}
    for split, path in split_paths.items():
        if path:
            out_metrics[split] = evaluate_model(model, load_dataset(path), split)
    test_metrics = out_metrics.get("test", {})
    underpowered = (
        len(x) < int(args.min_examples)
        or model is None
        or test_metrics.get("num_examples", 0) < max(1, int(args.min_examples * 0.1))
        or all(test_metrics.get(f"{target}_auroc") is None for target in ["hit", "contract_positive", "negative_progress"])
    )
    out_metrics["status"] = "CONTRACT_MODEL_UNDERPOWERED" if underpowered else "ok"
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_metrics).open("w", encoding="utf-8") as fh:
        json.dump(out_metrics, fh, indent=2, sort_keys=True)
    if args.out_report:
        write_report(args.out_report, out_metrics)
    print(json.dumps(out_metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

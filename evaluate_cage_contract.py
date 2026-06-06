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
from cage.contract_model import auprc, auroc, load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a minimal offline CAGE contract predictor.")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--test_path", default="")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--out_report", default="")
    return parser.parse_args()


def brier(y, pred):
    import numpy as np

    return float(np.mean((pred - y) ** 2)) if len(y) else None


def calibration_table(y, pred, bins: int = 5):
    import numpy as np

    rows = []
    if len(y) == 0:
        return rows
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred >= lo) & (pred <= hi if hi == 1.0 else pred < hi)
        if np.any(mask):
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


def write_report(path: str, metrics: dict) -> None:
    lines = [
        "# CAGE Contract Model Evaluation",
        "",
        f"- status: {metrics.get('status')}",
        f"- dataset_path: `{metrics.get('dataset_path')}`",
        f"- model_path: `{metrics.get('model_path')}`",
        "",
        "| label | AUROC | AUPRC | Brier | d_phi baseline AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in ["hit", "contract_positive", "negative_progress"]:
        lines.append(
            f"| {target} | {fmt(metrics.get(f'{target}_auroc'))} | {fmt(metrics.get(f'{target}_auprc'))} | "
            f"{fmt(metrics.get(f'{target}_brier'))} | {fmt(metrics.get(f'{target}_d_phi_baseline_auroc'))} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    return "NA" if value is None else f"{float(value):.3f}"


def main() -> int:
    args = parse_args()
    dataset_path = args.test_path or args.dataset_path
    if not dataset_path:
        raise SystemExit("--test_path or --dataset_path is required")
    dataset = ClosedLoopContractDataset.from_jsonl(dataset_path)
    x, labels = dataset.feature_matrix()
    model = load_model(args.model_path)
    metrics = {"num_examples": int(len(x)), "dataset_path": dataset_path, "model_path": args.model_path}
    import numpy as np

    d_phi = x[:, -1] if len(x) and x.shape[1] else np.empty((len(x),), dtype=np.float32)
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
    metrics["status"] = (
        "CONTRACT_MODEL_UNDERPOWERED"
        if len(x) == 0 or all(metrics.get(f"{target}_auroc") is None for target in ["hit", "contract_positive", "negative_progress"])
        else "ok"
    )
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_metrics).open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    if args.out_report:
        write_report(args.out_report, metrics)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

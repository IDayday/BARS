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
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_metrics", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = ClosedLoopContractDataset.from_jsonl(args.dataset_path)
    x, labels = dataset.feature_matrix()
    model = load_model(args.model_path)
    metrics = {"num_examples": int(len(x))}
    for target in sorted(model.weights):
        y = labels.get(target)
        if y is None or len(y) != len(x):
            continue
        pred = model.predict_proba(x, target)
        metrics[f"{target}_auroc"] = auroc(y, pred)
        metrics[f"{target}_auprc"] = auprc(y, pred)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_metrics).open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

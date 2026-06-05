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
from cage.contract_model import save_model, train_contract_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal offline linear contract predictor on CLP contract labels.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out_model", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = ClosedLoopContractDataset.from_jsonl(args.dataset_path)
    x, labels = dataset.feature_matrix()
    model, metrics = train_contract_model(x, labels, epochs=args.epochs, lr=args.lr)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    if model is not None:
        save_model(args.out_model, model)
        metrics["out_model"] = args.out_model
    else:
        metrics["out_model"] = None
    with Path(args.out_metrics).open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

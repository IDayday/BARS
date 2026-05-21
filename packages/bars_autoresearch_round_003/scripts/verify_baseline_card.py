#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "env",
    "suite",
    "algorithm",
    "baseline_role",
    "exact_public_target_available",
    "public_source",
    "public_metric",
    "public_mean",
    "public_std",
    "public_eval_protocol",
    "required_train_steps",
    "required_batch_size",
    "required_hyperparameters",
    "official_checkpoint_available",
    "official_tdr_available",
    "official_graph_available",
    "we_used",
    "official_eval_score",
    "bars_adapter_score",
    "adapter_gap_pp",
    "lower_bound",
    "certification_status",
}


def load_cards(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    cards = load_cards(Path(args.path))
    errors = []
    for i, card in enumerate(cards):
        missing = sorted(REQUIRED_KEYS.difference(card))
        if missing:
            errors.append({"index": i, "env": card.get("env"), "algorithm": card.get("algorithm"), "missing": missing})
        if card.get("algorithm") == "GAS" and card.get("required_train_steps") in (None, "", 0):
            errors.append({"index": i, "env": card.get("env"), "algorithm": "GAS", "missing": ["required_train_steps"]})
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps({"ok": True, "cards": len(cards)}, sort_keys=True))


if __name__ == "__main__":
    main()

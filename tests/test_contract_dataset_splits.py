from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_contract_dataset_splits import grouped_split, load_jsonl, split_stats, underpowered_reason, write_jsonl


def make_row(env: str, seed: int, task_id: int, variant: str, idx: int) -> dict:
    return {
        "env_name": env,
        "seed": seed,
        "task_id": task_id,
        "variant_source": variant,
        "phi_s": [float(idx), 0.0],
        "phi_g": [float(idx), 1.0],
        "hit": idx % 2 == 0,
        "label_hit": idx % 2 == 0,
        "label_contract_positive": idx % 3 == 0,
        "label_negative_progress": idx % 5 == 0,
        "final_phase": idx % 7 == 0,
        "recovery_candidate": idx % 11 == 0,
    }


def test_grouped_split_does_not_split_group_keys_across_splits():
    rows = []
    for task_id in range(8):
        for j in range(3):
            rows.append(make_row("antmaze", 42, task_id, "gas" if task_id % 2 == 0 else "cage", task_id * 3 + j))
    splits = grouped_split(
        rows,
        group_keys=["env_name", "seed", "task_id", "variant_source"],
        train_frac=0.5,
        val_frac=0.25,
        seed=0,
    )
    seen = {}
    for split_name, split_rows in splits.items():
        for row in split_rows:
            key = (row["env_name"], row["seed"], row["task_id"], row["variant_source"])
            assert key not in seen or seen[key] == split_name
            seen[key] = split_name
    assert all(splits[name] for name in ["train", "val", "test"])


def test_split_stats_and_underpowered_flags(tmp_path: Path):
    rows = [make_row("antmaze", 42, i, "gas", i) for i in range(6)]
    path = tmp_path / "data.jsonl"
    write_jsonl(path, rows)
    loaded = load_jsonl(path)
    assert len(loaded) == 6
    stats = split_stats(loaded)
    assert stats["num_examples"] == 6
    assert 0.0 <= stats["hit_rate"] <= 1.0
    splits = {"train": loaded[:4], "val": loaded[4:5], "test": loaded[5:]}
    reasons = underpowered_reason(splits, min_examples=100)
    assert "num_examples<100" in reasons

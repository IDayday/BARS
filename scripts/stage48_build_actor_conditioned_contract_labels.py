#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        path = path / "actor_augmented_contract_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def _finite_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return vals.fillna(default)


def _threshold_from_train_positive(env_df: pd.DataFrame, q: float, fallback_q: float) -> float:
    train = env_df[env_df["split"].astype(str) == "train"]
    pos_train = train[train["label_reach_base"] == 1]
    if len(pos_train):
        return float(pos_train["actor_action_mse"].quantile(q))
    labeled_train = train[train["label_reach_base"] >= 0]
    if len(labeled_train):
        return float(labeled_train["actor_action_mse"].quantile(fallback_q))
    return float(env_df["actor_action_mse"].quantile(fallback_q))


def build_labels(args: argparse.Namespace) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    for item in args.inputs:
        df = _read_table(Path(item))
        df["source_path"] = str(item)
        frames.append(df)
    if not frames:
        raise RuntimeError("No input actor-augmented contract datasets")
    rows = pd.concat(frames, ignore_index=True)
    required = {"env", "split", "label_reach", "actor_action_mse"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows["label_reach_base"] = pd.to_numeric(rows["label_reach"], errors="coerce").fillna(-1).astype(int)
    if "label_weight" in rows:
        rows["label_weight_base"] = _finite_numeric(rows["label_weight"], 1.0)
    else:
        rows["label_weight_base"] = 1.0
    rows["actor_action_mse"] = _finite_numeric(rows["actor_action_mse"], 0.0)

    out_frames: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for env, env_df in rows.groupby("env", sort=True):
        env_df = env_df.copy()
        threshold = _threshold_from_train_positive(
            env_df,
            q=float(args.positive_agree_quantile),
            fallback_q=float(args.fallback_quantile),
        )
        temperature = float(args.mse_temperature)
        if temperature <= 0:
            train_pos = env_df[(env_df["split"].astype(str) == "train") & (env_df["label_reach_base"] == 1)]
            if len(train_pos):
                temperature = max(float(train_pos["actor_action_mse"].median()), 1e-6)
            else:
                temperature = max(float(env_df["actor_action_mse"].median()), 1e-6)

        base_label = env_df["label_reach_base"].to_numpy(np.int64)
        mse = env_df["actor_action_mse"].to_numpy(np.float32)
        demote = (base_label == 1) & (mse > threshold)
        actor_label = base_label.copy()
        if bool(args.demote_disagreeing_positives):
            actor_label[demote] = 0

        agree_weight = np.exp(-mse / max(float(temperature), 1e-6)).astype(np.float32)
        label_weight = env_df["label_weight_base"].to_numpy(np.float32).copy()
        positive_mask = actor_label == 1
        label_weight[positive_mask] *= np.clip(agree_weight[positive_mask], float(args.min_positive_weight), 1.0)
        if bool(args.upweight_actor_hard_negatives):
            label_weight[demote] *= float(args.hard_negative_weight)

        reason = np.full(len(env_df), "base_label_kept", dtype=object)
        reason[demote] = "demoted_positive_high_actor_mse"
        reason[base_label < 0] = "unlabeled_kept"

        env_df["actor_conditioned_label_reach"] = actor_label
        env_df["actor_conditioned_label_weight"] = label_weight
        env_df["actor_agree_weight"] = agree_weight
        env_df["actor_mse_threshold"] = float(threshold)
        env_df["actor_mse_temperature"] = float(temperature)
        env_df["actor_demoted_positive"] = demote.astype(np.int32)
        env_df["actor_label_reason"] = reason

        # Keep compatibility with Stage45 trainer, which reads label_reach and
        # label_weight. The original labels are preserved as *_base columns.
        env_df["label_reach"] = env_df["actor_conditioned_label_reach"]
        env_df["label_weight"] = env_df["actor_conditioned_label_weight"]

        labeled_base = base_label >= 0
        labeled_actor = actor_label >= 0
        summaries[str(env)] = {
            "rows": int(len(env_df)),
            "threshold_from_train_positive_quantile": float(args.positive_agree_quantile),
            "actor_mse_threshold": float(threshold),
            "actor_mse_temperature": float(temperature),
            "base_labeled_rows": int(labeled_base.sum()),
            "actor_labeled_rows": int(labeled_actor.sum()),
            "base_positive_rate": float((base_label[labeled_base] == 1).mean()) if labeled_base.any() else 0.0,
            "actor_conditioned_positive_rate": float((actor_label[labeled_actor] == 1).mean()) if labeled_actor.any() else 0.0,
            "num_demoted_positives": int(demote.sum()),
            "demoted_positive_rate_among_base_positives": float(demote.sum() / max(int((base_label == 1).sum()), 1)),
            "actor_action_mse_mean": float(env_df["actor_action_mse"].mean()),
            "actor_action_mse_median": float(env_df["actor_action_mse"].median()),
        }
        out_frames.append(env_df)

    out = pd.concat(out_frames, ignore_index=True)
    out_path = Path(args.out)
    _write_table(out, out_path)
    summary: dict[str, Any] = {
        "offline_only": True,
        "uses_environment_rollout": False,
        "inputs": list(args.inputs),
        "output_csv": str(out_path),
        "rows": int(len(out)),
        "demote_disagreeing_positives": bool(args.demote_disagreeing_positives),
        "positive_agree_quantile": float(args.positive_agree_quantile),
        "fallback_quantile": float(args.fallback_quantile),
        "min_positive_weight": float(args.min_positive_weight),
        "upweight_actor_hard_negatives": bool(args.upweight_actor_hard_negatives),
        "hard_negative_weight": float(args.hard_negative_weight),
        "per_env": summaries,
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build actor-conditioned offline contract labels from Stage47 actor "
            "agreement features. This is an offline-only label transform."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--positive-agree-quantile", type=float, default=0.75)
    parser.add_argument("--fallback-quantile", type=float, default=0.50)
    parser.add_argument("--mse-temperature", type=float, default=0.0)
    parser.add_argument("--min-positive-weight", type=float, default=0.25)
    parser.add_argument("--demote-disagreeing-positives", type=int, default=1)
    parser.add_argument("--upweight-actor-hard-negatives", type=int, default=1)
    parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    args = parser.parse_args()
    build_labels(args)


if __name__ == "__main__":
    main()


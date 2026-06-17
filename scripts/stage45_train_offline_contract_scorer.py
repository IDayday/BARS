#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "phi_dist_pair",
    "edge_phi_dist",
    "node_u_dist",
    "node_v_dist",
    "local_support",
    "same_traj_support",
    "edge_forward",
    "h_exec",
    "way_steps",
]


def _load_inputs(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            path = path / "offline_contract_pairs.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        df["source_path"] = str(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("No input contract datasets")
    return pd.concat(frames, ignore_index=True)


def _feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name in feature_names:
        if name not in df:
            raise KeyError(f"Missing feature column: {name}")
        vals = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[name] = vals.fillna(vals.median() if vals.notna().any() else 0.0)
    if "edge_source" in df:
        for value in sorted(str(x) for x in df["edge_source"].dropna().unique()):
            out[f"edge_source={value}"] = (df["edge_source"].astype(str) == value).astype(np.float32)
    return out


def _metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    y = np.asarray(y, dtype=np.float32)
    prob = np.asarray(prob, dtype=np.float32)
    pred = (prob >= 0.5).astype(np.float32)
    out["accuracy"] = float((pred == y).mean()) if len(y) else 0.0
    out["positive_rate"] = float(y.mean()) if len(y) else 0.0
    out["predicted_positive_rate"] = float(pred.mean()) if len(y) else 0.0
    out["mean_prob"] = float(prob.mean()) if len(prob) else 0.0
    out["brier"] = float(np.mean((prob - y) ** 2)) if len(y) else 0.0
    try:
        from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

        out["average_precision"] = float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else 0.0
        out["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else 0.0
        out["log_loss"] = float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        pass
    return out


def _edge_aggregate_metrics(df: pd.DataFrame, prob: np.ndarray) -> dict[str, float]:
    if "edge_id" not in df:
        return {}
    tmp = df[["env", "edge_id", "label_reach"]].copy()
    tmp["prob"] = prob
    grouped = tmp.groupby(["env", "edge_id"], dropna=False).agg(label=("label_reach", "max"), prob=("prob", "mean"))
    if grouped.empty:
        return {}
    return {f"edge_{k}": v for k, v in _metrics(grouped["label"].to_numpy(), grouped["prob"].to_numpy()).items()}


def _rank_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    out: dict[str, float] = {}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y)) > 1:
            out["average_precision"] = float(average_precision_score(y, score))
            out["roc_auc"] = float(roc_auc_score(y, score))
        else:
            out["average_precision"] = 0.0
            out["roc_auc"] = 0.0
    except Exception:
        pass
    return out


def _baseline_rank_metrics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    y = df["label_reach"].to_numpy(np.float32)
    baselines: dict[str, np.ndarray] = {}
    if "local_support" in df:
        baselines["local_support"] = pd.to_numeric(df["local_support"], errors="coerce").fillna(0).to_numpy(np.float32)
    if "same_traj_support" in df:
        baselines["same_traj_support"] = pd.to_numeric(df["same_traj_support"], errors="coerce").fillna(0).to_numpy(np.float32)
    if "phi_dist_pair" in df:
        baselines["neg_phi_dist_pair"] = -pd.to_numeric(df["phi_dist_pair"], errors="coerce").fillna(0).to_numpy(np.float32)
    if "edge_phi_dist" in df:
        baselines["neg_edge_phi_dist"] = -pd.to_numeric(df["edge_phi_dist"], errors="coerce").fillna(0).to_numpy(np.float32)
    if "local_support" in df and "phi_dist_pair" in df:
        support = pd.to_numeric(df["local_support"], errors="coerce").fillna(0).to_numpy(np.float32)
        dist = pd.to_numeric(df["phi_dist_pair"], errors="coerce").fillna(0).to_numpy(np.float32)
        baselines["support_minus_distance"] = support - 0.01 * dist
    return {name: _rank_metrics(y, score) for name, score in baselines.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    all_df = _load_inputs(args.inputs)
    all_df["label_reach"] = pd.to_numeric(all_df["label_reach"], errors="coerce").astype(int)
    df = all_df[all_df["label_reach"] >= 0].copy()
    if df.empty:
        raise RuntimeError("No labeled rows after filtering label_reach >= 0")
    features = list(args.features or DEFAULT_FEATURES)
    x_df = _feature_frame(df, features)
    y = df["label_reach"].to_numpy(np.int64)
    sample_weight = None
    if bool(args.use_label_weight) and "label_weight" in df:
        sample_weight = pd.to_numeric(df["label_weight"], errors="coerce").fillna(1.0).to_numpy(np.float32)
        sample_weight = np.clip(sample_weight, 0.0, np.inf)
    split = df["split"].astype(str).to_numpy()

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    if train_mask.sum() == 0:
        raise RuntimeError("No train rows")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    class_weight = "balanced" if args.class_weight_balanced else None
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=int(args.max_iter),
                    class_weight=class_weight,
                    C=float(args.c),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["logreg__sample_weight"] = sample_weight[train_mask]
    model.fit(x_df.loc[train_mask], y[train_mask], **fit_kwargs)
    all_x_df = _feature_frame(all_df, features)
    all_x_df = all_x_df.reindex(columns=x_df.columns, fill_value=0.0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump({"model": model, "features": list(x_df.columns)}, out / "contract_logistic.joblib")
    except Exception:
        pass

    prob_labeled = model.predict_proba(x_df)[:, 1]
    prob_all_rows = model.predict_proba(all_x_df)[:, 1]
    scored_cols = [
        "env",
        "seed",
        "edge_id",
        "edge_u",
        "edge_v",
        "edge_forward",
        "edge_source",
        "sample_kind",
        "split",
        "label_reach",
        "local_support",
        "same_traj_support",
        "phi_dist_pair",
        "edge_phi_dist",
        "node_u_dist",
        "node_v_dist",
    ]
    if "label_weight" in all_df:
        scored_cols.append("label_weight")
    for col in features:
        if col in all_df and col not in scored_cols:
            scored_cols.append(col)
    scored = all_df[scored_cols].copy()
    scored["contract_prob"] = prob_all_rows
    scored.to_csv(out / "contract_scored_rows.csv", index=False)
    try:
        scored.to_parquet(out / "contract_scored_rows.parquet", index=False)
    except Exception:
        pass

    metrics: dict[str, Any] = {
        "offline_only": True,
        "uses_environment_rollout": False,
        "inputs": args.inputs,
        "num_rows": int(len(df)),
        "num_all_rows_scored": int(len(all_df)),
        "features": list(x_df.columns),
        "class_weight_balanced": bool(args.class_weight_balanced),
        "uses_label_weight": bool(sample_weight is not None),
        "splits": {},
    }
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        if int(mask.sum()) == 0:
            continue
        metrics["splits"][name] = {
            "rows": int(mask.sum()),
            **_metrics(y[mask], prob_labeled[mask]),
            **_edge_aggregate_metrics(df.loc[mask], prob_labeled[mask]),
            "rank_baselines": _baseline_rank_metrics(df.loc[mask]),
        }
    per_env: dict[str, Any] = {}
    for env_name, env_df in df.groupby("env"):
        idx = env_df.index.to_numpy()
        env_prob = prob_labeled[df.index.get_indexer(idx)]
        per_env[str(env_name)] = {
            "rows": int(len(env_df)),
            **_metrics(env_df["label_reach"].to_numpy(), env_prob),
            **_edge_aggregate_metrics(env_df, env_prob),
            "rank_baselines": _baseline_rank_metrics(env_df),
        }
    metrics["per_env"] = per_env

    # Coefficients are useful for debugging whether the model just learned a
    # distance heuristic.
    try:
        lr = model.named_steps["logreg"]
        metrics["coefficients"] = {
            name: float(value) for name, value in sorted(zip(x_df.columns, lr.coef_[0]), key=lambda item: abs(item[1]), reverse=True)
        }
        metrics["intercept"] = float(lr.intercept_[0])
    except Exception:
        pass

    (out / "contract_scorer_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight offline contract scorer on Stage45 contract pairs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Contract dataset directories or CSV/parquet files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--class-weight-balanced", type=int, default=1)
    parser.add_argument("--use-label-weight", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .bridge_graph import BRIDGE_EDGE_TYPES, RISKY_EDGE_TYPES, node_phis


FEATURE_COLUMNS = [
    "phi_dist",
    "tdr_cost",
    "local_support",
    "same_traj_support",
    "bridge_score",
    "bottleneck_score",
    "gas_weight",
    "temporal_cost",
    "edge_type_safe_local",
    "edge_type_same_traj_temporal",
    "edge_type_gas_cross",
    "edge_type_aggressive_tdr_bridge",
    "edge_type_bottleneck_bridge",
]


@dataclass
class BridgeDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    feature_columns: list[str]


def make_bridge_table(edge_exec: pd.DataFrame, bridge_table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = edge_exec.copy()
    if bridge_table is not None and len(bridge_table) and "edge_id" in bridge_table:
        cols = [c for c in bridge_table.columns if c not in df.columns or c == "edge_id"]
        df = df.merge(bridge_table[cols].drop_duplicates("edge_id"), on="edge_id", how="left", suffixes=("", "_bridge"))
    if "edge_type" not in df:
        df["edge_type"] = "unknown"
    for c in ["phi_dist", "tdr_cost", "local_support", "same_traj_support", "bridge_score", "bottleneck_score", "gas_weight", "temporal_cost"]:
        if c not in df:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "success" not in df:
        raise ValueError("edge execution labels must include a success column")
    df["label"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(np.float32)
    for et in ["safe_local", "same_traj_temporal", "gas_cross", "aggressive_tdr_bridge", "bottleneck_bridge"]:
        df[f"edge_type_{et}"] = (df["edge_type"].astype(str) == et).astype(np.float32)
    df["is_selected_bridge"] = df["edge_type"].astype(str).isin(BRIDGE_EDGE_TYPES).astype(int)
    df["is_risky_edge"] = df["edge_type"].astype(str).isin(RISKY_EDGE_TYPES).astype(int)
    return df


def split_bridge_dataset(df: pd.DataFrame, val_frac: float = 0.25, seed: int = 0) -> BridgeDataset:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0 and len(val_idx):
        train_idx = val_idx
    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy() if len(val_idx) else df.iloc[train_idx].copy()
    x_train = train[FEATURE_COLUMNS].to_numpy(np.float32)
    y_train = train["label"].to_numpy(np.float32)
    x_val = val[FEATURE_COLUMNS].to_numpy(np.float32)
    y_val = val["label"].to_numpy(np.float32)
    return BridgeDataset(x_train, y_train, x_val, y_val, train, val, FEATURE_COLUMNS.copy())


def load_bridge_dataset(edge_exec_csv: str | Path, bridge_table_csv: str | Path | None = None, seed: int = 0) -> BridgeDataset:
    edge_exec = pd.read_csv(edge_exec_csv)
    bridge_table = pd.read_csv(bridge_table_csv) if bridge_table_csv and Path(bridge_table_csv).exists() else None
    df = make_bridge_table(edge_exec, bridge_table)
    return split_bridge_dataset(df, seed=seed)

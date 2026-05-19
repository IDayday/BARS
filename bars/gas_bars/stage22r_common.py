from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def parse_csv_list(spec: str | Iterable[str]) -> list[str]:
    if isinstance(spec, str):
        return [x.strip() for x in spec.split(",") if x.strip()]
    return [str(x).strip() for x in spec if str(x).strip()]


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if len(df) == 0:
        return "_No rows._"
    view = df.head(max_rows) if max_rows else df
    try:
        return view.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + view.to_csv(index=False).strip() + "\n```"


def read_all_eval(eval_root: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(eval_root).rglob("eval.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if len(df) == 0:
            continue
        df["eval_path"] = str(path)
        if "fallback_mode" not in df.columns:
            df["fallback_mode"] = path.parent.name.replace("fallback_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_eval(df: pd.DataFrame, envs: list[str], seeds: list[int]) -> pd.DataFrame:
    out = df
    if envs and "env" in out.columns:
        out = out[out["env"].astype(str).isin(envs)]
    if seeds and "seed" in out.columns:
        out = out[out["seed"].astype(int).isin(seeds)]
    return out.copy()


def load_edge_scores(artifact_root: str | Path, env: str, seed: int) -> pd.DataFrame:
    path = Path(artifact_root) / env / f"seed{seed}" / "edge_scores.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_boundary_scores(artifact_root: str | Path, env: str, seed: int) -> pd.DataFrame:
    path = Path(artifact_root) / env / f"seed{seed}" / "boundary_scores.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def parse_edge_ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    out = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            continue
    return out


def edge_lookup(edge_scores: pd.DataFrame) -> dict[int, dict[str, object]]:
    if len(edge_scores) == 0 or "edge_id" not in edge_scores.columns:
        return {}
    return {int(r.edge_id): r._asdict() for r in edge_scores.itertuples(index=False)}


def boundary_lookup(boundary_scores: pd.DataFrame) -> dict[tuple[int, int], dict[str, object]]:
    if len(boundary_scores) == 0:
        return {}
    rows: dict[tuple[int, int], dict[str, object]] = {}
    for r in boundary_scores.itertuples(index=False):
        rows[(int(r.prev_edge_id), int(r.next_edge_id))] = r._asdict()
    return rows


def path_edge_metrics(
    edge_ids: list[int],
    edge_scores: pd.DataFrame,
    boundary_scores: Optional[pd.DataFrame] = None,
    fallback_boundary_cost: float = -math.log(0.1),
) -> dict[str, float]:
    edges = edge_lookup(edge_scores)
    bmap = boundary_lookup(boundary_scores if boundary_scores is not None else pd.DataFrame())
    return path_edge_metrics_from_lookup(edge_ids, edges, bmap, fallback_boundary_cost)


def path_edge_metrics_from_lookup(
    edge_ids: list[int],
    edges: dict[int, dict[str, object]],
    bmap: dict[tuple[int, int], dict[str, object]],
    fallback_boundary_cost: float = -math.log(0.1),
) -> dict[str, float]:
    known_rows = [edges[e] for e in edge_ids if e in edges]
    known = len(known_rows)
    edge_count = len(edge_ids)
    p_vals = np.asarray([float(r.get("p_exec", np.nan)) for r in known_rows], dtype=np.float64)
    r_vals = np.asarray([float(r.get("r_exec", 0.0)) for r in known_rows], dtype=np.float64)
    local_vals = np.asarray([float(r.get("local_support", 0.0)) for r in known_rows], dtype=np.float64)
    same_vals = np.asarray([float(r.get("same_traj_support", 0.0)) for r in known_rows], dtype=np.float64)
    virtual_edges = edge_count - known
    missing_pairs = 0
    virtual_pairs = 0
    unsupported_pairs = 0
    boundary_cost = 0.0
    psi_vals = []
    for a, b in zip(edge_ids[:-1], edge_ids[1:]):
        pair = bmap.get((a, b))
        if pair is None:
            missing_pairs += 1
            boundary_cost += fallback_boundary_cost
            if a not in edges or b not in edges:
                virtual_pairs += 1
            continue
        cost = float(pair.get("boundary_cost", fallback_boundary_cost))
        boundary_cost += cost
        if "psi" in pair:
            psi_vals.append(float(pair["psi"]))
        support_type = str(pair.get("support_type", ""))
        if "unsupported" in support_type:
            unsupported_pairs += 1
    pair_count = max(edge_count - 1, 0)
    return {
        "path_edges": float(edge_count),
        "known_edges": float(known),
        "virtual_edges": float(virtual_edges),
        "known_edge_rate": known / max(edge_count, 1),
        "p_exec_mean": float(np.nanmean(p_vals)) if len(p_vals) else np.nan,
        "p_exec_min": float(np.nanmin(p_vals)) if len(p_vals) else np.nan,
        "r_exec_static": float(np.nansum(r_vals)) if len(r_vals) else 0.0,
        "local_support_rate": float(np.nanmean(local_vals > 0)) if len(local_vals) else np.nan,
        "unsupported_edge_rate": float(np.nanmean(local_vals <= 0)) if len(local_vals) else np.nan,
        "same_traj_support_mean": float(np.nanmean(same_vals)) if len(same_vals) else np.nan,
        "boundary_cost_static": float(boundary_cost),
        "missing_boundary_pairs": float(missing_pairs),
        "missing_boundary_pair_rate": missing_pairs / max(pair_count, 1),
        "virtual_boundary_pairs": float(virtual_pairs),
        "unsupported_boundary_pairs": float(unsupported_pairs),
        "unsupported_boundary_pair_rate": unsupported_pairs / max(pair_count, 1),
        "psi_mean": float(np.mean(psi_vals)) if psi_vals else np.nan,
    }


def add_path_metrics(df: pd.DataFrame, artifact_root: str | Path) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    rows = []
    cache: dict[tuple[str, int], tuple[dict[int, dict[str, object]], dict[tuple[int, int], dict[str, object]]]] = {}
    for r in df.itertuples(index=False):
        env = str(getattr(r, "env"))
        seed = int(getattr(r, "seed"))
        key = (env, seed)
        if key not in cache:
            cache[key] = (
                edge_lookup(load_edge_scores(artifact_root, env, seed)),
                boundary_lookup(load_boundary_scores(artifact_root, env, seed)),
            )
        edges, bmap = cache[key]
        metrics = path_edge_metrics_from_lookup(parse_edge_ids(getattr(r, "path_edge_ids", "")), edges, bmap)
        rows.append(metrics)
    metrics_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), metrics_df], axis=1)


def quantile_dict(values: pd.Series | np.ndarray, prefix: str) -> dict[str, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(np.float64)
    out: dict[str, float] = {}
    if len(arr) == 0:
        return out
    for q in (0.5, 0.6, 0.7, 0.8):
        out[f"{prefix}_q{int(q * 100)}"] = float(np.quantile(arr, q))
    return out

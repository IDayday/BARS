from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.int64)
    return ranks


def _top_clusters(df: pd.DataFrame, column: str, budget: int) -> list[int]:
    if budget <= 0:
        return []
    ranked = df.sort_values([column, "cluster"], ascending=[False, True], kind="mergesort")
    return ranked["cluster"].head(budget).astype(int).tolist()


def _core_plus_bottleneck(df: pd.DataFrame, budget: int) -> set[int]:
    if budget <= 0:
        return set()
    half = budget // 2
    selected: list[int] = []
    selected.extend(_top_clusters(df, "density", half))
    selected.extend(_top_clusters(df, "bottleneck_score", budget - half))
    out: list[int] = []
    seen: set[int] = set()
    for cluster in selected:
        if cluster not in seen:
            seen.add(cluster)
            out.append(cluster)

    density_order = _top_clusters(df, "density", len(df))
    bottleneck_order = _top_clusters(df, "bottleneck_score", len(df))
    di = 0
    bi = 0
    use_density = True
    while len(out) < min(budget, len(df)):
        source = density_order if use_density else bottleneck_order
        idx = di if use_density else bi
        if idx >= len(source):
            use_density = not use_density
            if di >= len(density_order) and bi >= len(bottleneck_order):
                break
            continue
        cluster = source[idx]
        if use_density:
            di += 1
        else:
            bi += 1
        use_density = not use_density
        if cluster not in seen:
            seen.add(cluster)
            out.append(cluster)
    return set(out[:budget])


def select_nodes(
    density_df: pd.DataFrame,
    bottleneck_df: pd.DataFrame,
    method: str,
    budget: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Select occupied clusters for the compressed option graph."""

    del seed
    method = method.lower()
    density = density_df[["cluster", "count", "density"]].copy()
    merged = density.merge(
        bottleneck_df[["cluster", "bottleneck_score"]],
        on="cluster",
        how="left",
    )
    merged["bottleneck_score"] = merged["bottleneck_score"].fillna(0.0)
    occupied = merged[merged["count"] > 0].copy().reset_index(drop=True)
    if occupied.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "selected",
                "selection_method",
                "density",
                "bottleneck_score",
                "rank_density",
                "rank_bottleneck",
            ]
        )

    occupied["rank_density"] = _rank_desc(occupied["density"].to_numpy(dtype=np.float64))
    occupied["rank_bottleneck"] = _rank_desc(
        occupied["bottleneck_score"].to_numpy(dtype=np.float64)
    )
    budget = min(int(budget), int(occupied.shape[0])) if method != "all" else int(occupied.shape[0])
    if method == "all":
        selected = set(occupied["cluster"].astype(int).tolist())
    elif method == "density":
        selected = set(_top_clusters(occupied, "density", budget))
    elif method == "bottleneck":
        selected = set(_top_clusters(occupied, "bottleneck_score", budget))
    elif method == "core_plus_bottleneck":
        selected = _core_plus_bottleneck(occupied, budget)
    else:
        raise ValueError(
            "method must be density, bottleneck, core_plus_bottleneck, or all; "
            f"got {method!r}"
        )

    occupied["selected"] = occupied["cluster"].astype(int).isin(selected)
    occupied["selection_method"] = method
    return occupied[
        [
            "cluster",
            "selected",
            "selection_method",
            "density",
            "bottleneck_score",
            "rank_density",
            "rank_bottleneck",
        ]
    ].sort_values("cluster", kind="mergesort")


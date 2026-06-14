from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from phase3e.risk_aware_planning import (
    RiskPlannerConfig,
    evaluate_planning_methods,
    summarize_planning_results,
)


@dataclass(frozen=True)
class RiskSweepConfig:
    planner_method: str
    risk_weight: float
    ood_weight: float
    incompat_weight: float
    uncertified_weight: float
    min_proxy_score: float
    min_heldout_support_lcb: float
    high_ood_threshold: float = 0.5
    high_incompat_threshold: float = 0.5

    def planner_config(self) -> RiskPlannerConfig:
        return RiskPlannerConfig(
            risk_weight=self.risk_weight,
            ood_weight=self.ood_weight,
            incompat_weight=self.incompat_weight,
            uncertified_weight=self.uncertified_weight,
            min_proxy_score=self.min_proxy_score,
            min_heldout_support_lcb=self.min_heldout_support_lcb,
            high_ood_threshold=self.high_ood_threshold,
            high_incompat_threshold=self.high_incompat_threshold,
        )


def _float_grid(values: Iterable[float]) -> list[float]:
    return [float(x) for x in values]


def make_sweep_configs(
    planner_methods: Iterable[str],
    risk_weights: Iterable[float],
    ood_weights: Iterable[float],
    incompat_weights: Iterable[float],
    uncertified_weights: Iterable[float],
    proxy_floors: Iterable[float],
    heldout_support_lcb_floors: Iterable[float],
    high_ood_threshold: float = 0.5,
    high_incompat_threshold: float = 0.5,
) -> list[RiskSweepConfig]:
    configs: list[RiskSweepConfig] = []
    for method in planner_methods:
        for risk_weight in _float_grid(risk_weights):
            for ood_weight in _float_grid(ood_weights):
                for incompat_weight in _float_grid(incompat_weights):
                    for uncertified_weight in _float_grid(uncertified_weights):
                        for proxy_floor in _float_grid(proxy_floors):
                            for support_floor in _float_grid(heldout_support_lcb_floors):
                                configs.append(
                                    RiskSweepConfig(
                                        planner_method=str(method),
                                        risk_weight=risk_weight,
                                        ood_weight=ood_weight,
                                        incompat_weight=incompat_weight,
                                        uncertified_weight=uncertified_weight,
                                        min_proxy_score=proxy_floor,
                                        min_heldout_support_lcb=support_floor,
                                        high_ood_threshold=float(high_ood_threshold),
                                        high_incompat_threshold=float(high_incompat_threshold),
                                    )
                                )
    return configs


def run_planner_sweep(
    edge_table: pd.DataFrame,
    path_queries: pd.DataFrame,
    sweep_configs: list[RiskSweepConfig],
    max_queries: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sweep_id, sweep_config in enumerate(sweep_configs):
        paths, graphs = evaluate_planning_methods(
            edge_table,
            path_queries,
            methods=[sweep_config.planner_method],
            config=sweep_config.planner_config(),
            max_queries=max_queries,
            seed=seed,
        )
        summary = summarize_planning_results(paths, graphs)
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row.update(asdict(sweep_config))
        row["sweep_id"] = int(sweep_id)
        row["method"] = f"{sweep_config.planner_method}_s{sweep_id:04d}"
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = add_derived_sweep_metrics(out)
    return mark_pareto_front(out)


def add_derived_sweep_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["coverage_cost_ratio"] = out["path_coverage"] / out["mean_base_path_cost"].replace(0, np.nan)
    out["risk_cleanliness_score"] = (
        out["mean_min_edge_proxy_score"].fillna(0.0)
        + out["mean_heldout_support_lcb"].fillna(0.0)
        + (1.0 - out["mean_uncertified_edge_fraction"].fillna(1.0))
        + (1.0 - out["mean_high_ood_edge_fraction"].fillna(1.0))
        + (1.0 - out["mean_high_incompat_edge_fraction"].fillna(1.0))
    ) / 5.0
    out["coverage_risk_score"] = out["path_coverage"].fillna(0.0) * out["risk_cleanliness_score"].fillna(0.0)
    return out


def _dominates(a: pd.Series, b: pd.Series, objectives: list[tuple[str, str]]) -> bool:
    better_or_equal = True
    strictly_better = False
    for col, direction in objectives:
        av = float(a[col]) if np.isfinite(float(a[col])) else (-np.inf if direction == "max" else np.inf)
        bv = float(b[col]) if np.isfinite(float(b[col])) else (-np.inf if direction == "max" else np.inf)
        if direction == "max":
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True
    return bool(better_or_equal and strictly_better)


def mark_pareto_front(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy().reset_index(drop=True)
    objectives = [
        ("path_coverage", "max"),
        ("mean_min_edge_proxy_score", "max"),
        ("mean_uncertified_edge_fraction", "min"),
        ("mean_base_path_cost", "min"),
    ]
    is_pareto = np.ones(out.shape[0], dtype=bool)
    for i in range(out.shape[0]):
        if not is_pareto[i]:
            continue
        for j in range(out.shape[0]):
            if i == j:
                continue
            if _dominates(out.iloc[j], out.iloc[i], objectives):
                is_pareto[i] = False
                break
    out["is_pareto"] = is_pareto
    return out


def select_recommended_config(
    sweep_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    min_coverage_ratio: float = 0.95,
    max_base_cost_increase: float = 0.2,
) -> dict[str, object]:
    if sweep_summary.empty:
        return {}
    baseline = baseline_summary.set_index("method").loc["support_shortest_path"]
    baseline_cov = float(baseline["path_coverage"])
    baseline_cost = float(baseline["mean_base_path_cost"])
    min_coverage = baseline_cov * float(min_coverage_ratio)
    max_cost = baseline_cost * (1.0 + float(max_base_cost_increase))
    candidates = sweep_summary[
        (sweep_summary["path_coverage"] >= min_coverage)
        & (sweep_summary["mean_base_path_cost"] <= max_cost)
    ].copy()
    if candidates.empty:
        candidates = sweep_summary.copy()
    candidates["recommendation_score"] = (
        candidates["coverage_risk_score"].fillna(0.0)
        + 0.5 * candidates["mean_min_edge_proxy_score"].fillna(0.0)
        - 0.25 * candidates["mean_uncertified_edge_fraction"].fillna(1.0)
        - 0.1
        * (
            candidates["mean_base_path_cost"].fillna(baseline_cost) / max(1e-12, baseline_cost)
        )
    )
    ranked = candidates.sort_values(
        ["recommendation_score", "path_coverage", "mean_min_edge_proxy_score"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    if ranked.empty:
        return {}
    return ranked.iloc[0].to_dict()

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegretGuardConfig:
    baseline_method: str = "augmented_loss_support_bottleneck_s03"
    max_final_val_ratio: float = 1.01
    max_direct_repair_ratio: float = 1.0
    max_planner_used_ratio: float = 0.99
    min_policy_support_ratio: float = 1.0
    allow_relaxed_improvement_fallback: bool = True
    relaxed_max_direct_repair_ratio: float = 1.0
    relaxed_max_planner_used_ratio: float = 1.0
    relaxed_min_policy_support_ratio: float = 1.0


RATIO_COLUMNS = [
    "final_val_action_mse_ratio_vs_baseline",
    "direct_repair_edge_mse_ratio_vs_baseline",
    "planner_used_repair_edge_mse_ratio_vs_baseline",
    "direct_repair_policy_support_score_ratio_vs_baseline",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _as_float_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _violation_reasons(row: pd.Series) -> str:
    reasons = []
    for flag, reason in [
        ("passes_final_val_guard", "final_val_regret"),
        ("passes_direct_repair_guard", "direct_repair_mse"),
        ("passes_planner_used_guard", "planner_used_repair_mse"),
        ("passes_policy_support_guard", "policy_support_score"),
    ]:
        if not bool(row.get(flag, False)):
            reasons.append(reason)
    return ";".join(reasons)


def annotate_regret_guard_candidates(
    comparisons: pd.DataFrame,
    config: RegretGuardConfig | None = None,
) -> pd.DataFrame:
    """Add guard metrics to a Phase 4M baseline-comparison table."""

    cfg = config or RegretGuardConfig()
    if comparisons.empty:
        return pd.DataFrame()
    out = comparisons.copy()
    for col in RATIO_COLUMNS:
        out[col] = _as_float_series(out, col)
    out["is_baseline"] = out["method"].astype(str) == cfg.baseline_method
    out["final_val_mse_regret"] = out["final_val_action_mse_ratio_vs_baseline"] - 1.0
    out["direct_repair_mse_improvement"] = 1.0 - out["direct_repair_edge_mse_ratio_vs_baseline"]
    out["planner_used_repair_mse_improvement"] = 1.0 - out["planner_used_repair_edge_mse_ratio_vs_baseline"]
    out["policy_support_score_gain"] = out["direct_repair_policy_support_score_ratio_vs_baseline"] - 1.0
    out["passes_final_val_guard"] = out["final_val_action_mse_ratio_vs_baseline"] <= float(cfg.max_final_val_ratio)
    out["passes_direct_repair_guard"] = out["direct_repair_edge_mse_ratio_vs_baseline"] <= float(
        cfg.max_direct_repair_ratio
    )
    out["passes_planner_used_guard"] = out["planner_used_repair_edge_mse_ratio_vs_baseline"] <= float(
        cfg.max_planner_used_ratio
    )
    out["passes_policy_support_guard"] = out["direct_repair_policy_support_score_ratio_vs_baseline"] >= float(
        cfg.min_policy_support_ratio
    )
    out["passes_relaxed_direct_repair_guard"] = out["direct_repair_edge_mse_ratio_vs_baseline"] <= float(
        cfg.relaxed_max_direct_repair_ratio
    )
    out["passes_relaxed_planner_used_guard"] = out["planner_used_repair_edge_mse_ratio_vs_baseline"] <= float(
        cfg.relaxed_max_planner_used_ratio
    )
    out["passes_relaxed_policy_support_guard"] = out["direct_repair_policy_support_score_ratio_vs_baseline"] >= float(
        cfg.relaxed_min_policy_support_ratio
    )
    out["guard_pass"] = (
        ~out["is_baseline"]
        & out["passes_final_val_guard"]
        & out["passes_direct_repair_guard"]
        & out["passes_planner_used_guard"]
        & out["passes_policy_support_guard"]
    )
    out["relaxed_guard_pass"] = (
        ~out["is_baseline"]
        & out["passes_final_val_guard"]
        & out["passes_relaxed_direct_repair_guard"]
        & out["passes_relaxed_planner_used_guard"]
        & out["passes_relaxed_policy_support_guard"]
    )
    out["num_guard_violations"] = (
        (~out["passes_final_val_guard"]).astype(int)
        + (~out["passes_direct_repair_guard"]).astype(int)
        + (~out["passes_planner_used_guard"]).astype(int)
        + (~out["passes_policy_support_guard"]).astype(int)
    )
    out["guard_violation_reasons"] = out.apply(_violation_reasons, axis=1)
    return out


def select_regret_guard_candidate(
    comparisons: pd.DataFrame,
    config: RegretGuardConfig | None = None,
) -> dict[str, Any]:
    """Select a guarded planner-relevant candidate from one comparison table.

    The selector is intentionally conservative. It only recommends a non-baseline
    method when all guards pass. If no candidate passes, it returns the baseline
    as a fallback and records why no planner-relevant method was selected.
    """

    cfg = config or RegretGuardConfig()
    annotated = annotate_regret_guard_candidates(comparisons, cfg)
    if annotated.empty:
        return {
            "selection_status": "no_rows",
            "selected_method": None,
            "selected_is_baseline": False,
            "num_candidates": 0,
            "num_guard_pass_candidates": 0,
        }
    candidates = annotated[~annotated["is_baseline"]].copy()
    passing = candidates[candidates["guard_pass"]].copy()
    relaxed = candidates[candidates["relaxed_guard_pass"]].copy()
    sort_cols = [
        "planner_used_repair_edge_mse_ratio_vs_baseline",
        "final_val_action_mse_ratio_vs_baseline",
        "direct_repair_edge_mse_ratio_vs_baseline",
        "direct_repair_policy_support_score_ratio_vs_baseline",
    ]
    ascending = [True, True, True, False]
    if not passing.empty:
        selected = passing.sort_values(sort_cols, ascending=ascending, kind="mergesort").iloc[0]
        status = "guard_pass"
    elif bool(cfg.allow_relaxed_improvement_fallback) and not relaxed.empty:
        selected = relaxed.sort_values(sort_cols, ascending=ascending, kind="mergesort").iloc[0]
        status = "relaxed_guard_pass"
    else:
        baseline = annotated[annotated["is_baseline"]]
        if not baseline.empty:
            selected = baseline.iloc[0]
            status = "fallback_baseline_no_guard_pass"
        elif not candidates.empty:
            selected = candidates.sort_values(
                [
                    "num_guard_violations",
                    "final_val_action_mse_ratio_vs_baseline",
                    "planner_used_repair_edge_mse_ratio_vs_baseline",
                    "direct_repair_edge_mse_ratio_vs_baseline",
                ],
                ascending=[True, True, True, True],
                kind="mergesort",
            ).iloc[0]
            status = "fallback_least_bad_no_baseline"
        else:
            selected = annotated.iloc[0]
            status = "fallback_only_row"

    row = selected.to_dict()
    row.update(
        {
            "selection_status": status,
            "selected_method": selected.get("method"),
            "selected_is_baseline": bool(selected.get("is_baseline", False)),
            "num_candidates": int(candidates.shape[0]),
            "num_guard_pass_candidates": int(passing.shape[0]),
            "num_relaxed_guard_pass_candidates": int(relaxed.shape[0]),
        }
    )
    return row


def collect_phase4m_guard_inputs(result_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(result_root).expanduser()
    all_rows: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/*/phase4m_vs_baseline.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        run_dir = path.parent
        dataset_key = run_dir.parent.name
        run_name = run_dir.name
        df.insert(0, "run_dir", str(run_dir))
        df.insert(1, "dataset_key", dataset_key)
        df.insert(2, "run_name", run_name)
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()
    combined = pd.concat(all_rows, ignore_index=True)
    run_keys = combined[["run_dir", "dataset_key", "run_name"]].drop_duplicates().reset_index(drop=True)
    return combined, run_keys


def run_regret_guard_selection(
    result_root: str | Path = "results/phase4m",
    config: RegretGuardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = config or RegretGuardConfig()
    combined, run_keys = collect_phase4m_guard_inputs(result_root)
    if combined.empty:
        payload = {
            "phase": "Phase 4O",
            "title": "Planner-Relevance Regret-Guard Selection",
            "num_runs": 0,
            "note": "Offline supervised selector; not rollout success.",
        }
        return pd.DataFrame(), pd.DataFrame(), payload

    annotated_rows: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    for _, run in run_keys.iterrows():
        mask = (combined["run_dir"] == run["run_dir"]) & (combined["run_name"] == run["run_name"])
        table = combined[mask].copy()
        annotated = annotate_regret_guard_candidates(table, cfg)
        annotated_rows.append(annotated)
        selected_rows.append(select_regret_guard_candidate(table, cfg))
    annotated_table = pd.concat(annotated_rows, ignore_index=True)
    selection_table = pd.DataFrame(selected_rows)
    nonbaseline = selection_table[~selection_table.get("selected_is_baseline", False).astype(bool)].copy()
    payload = {
        "phase": "Phase 4O",
        "title": "Planner-Relevance Regret-Guard Selection",
        "config": cfg.__dict__,
        "num_runs": int(selection_table.shape[0]),
        "num_runs_with_guard_pass": int((selection_table["selection_status"] == "guard_pass").sum()),
        "num_runs_with_relaxed_guard_pass": int((selection_table["selection_status"] == "relaxed_guard_pass").sum()),
        "num_runs_fallback_baseline": int(selection_table["selected_is_baseline"].astype(bool).sum()),
        "selected_methods": selection_table.get("selected_method", pd.Series(dtype=object)).astype(str).tolist(),
        "mean_selected_final_val_ratio": float(
            pd.to_numeric(selection_table.get("final_val_action_mse_ratio_vs_baseline"), errors="coerce").mean()
        ),
        "mean_selected_direct_repair_ratio": float(
            pd.to_numeric(selection_table.get("direct_repair_edge_mse_ratio_vs_baseline"), errors="coerce").mean()
        ),
        "mean_selected_planner_used_repair_ratio": float(
            pd.to_numeric(selection_table.get("planner_used_repair_edge_mse_ratio_vs_baseline"), errors="coerce").mean()
        ),
        "mean_nonbaseline_selected_final_val_ratio": float(
            pd.to_numeric(nonbaseline.get("final_val_action_mse_ratio_vs_baseline"), errors="coerce").mean()
        )
        if not nonbaseline.empty
        else None,
        "note": "Offline supervised selector; not rollout success.",
    }
    return annotated_table, selection_table, payload


def render_regret_guard_markdown(payload: dict[str, Any], selection: pd.DataFrame) -> str:
    lines = [
        "# Phase 4O Planner-Relevance Regret-Guard Selection",
        "",
        "This phase turns the Phase 4N manual Scene H10 choice into a reusable",
        "offline supervised model-selection guard. It does not train a new policy,",
        "run an environment rollout, or add unsupported graph edges.",
        "",
        "## Guard",
        "",
    ]
    config = payload.get("config", {})
    for key in [
        "max_final_val_ratio",
        "max_direct_repair_ratio",
        "max_planner_used_ratio",
        "min_policy_support_ratio",
        "allow_relaxed_improvement_fallback",
        "relaxed_max_direct_repair_ratio",
        "relaxed_max_planner_used_ratio",
        "relaxed_min_policy_support_ratio",
    ]:
        lines.append(f"- `{key}`: `{config.get(key)}`")
    lines.extend(["", "## Aggregate", ""])
    for key in [
        "num_runs",
        "num_runs_with_guard_pass",
        "num_runs_with_relaxed_guard_pass",
        "num_runs_fallback_baseline",
        "mean_selected_final_val_ratio",
        "mean_selected_direct_repair_ratio",
        "mean_selected_planner_used_repair_ratio",
    ]:
        lines.append(f"- `{key}`: `{payload.get(key)}`")
    lines.extend(["", "## Selected Runs", ""])
    if selection.empty:
        lines.append("No Phase 4M comparison tables were found.")
    else:
        cols = [
            "dataset_key",
            "run_name",
            "selected_method",
            "selection_status",
            "final_val_action_mse_ratio_vs_baseline",
            "direct_repair_edge_mse_ratio_vs_baseline",
            "planner_used_repair_edge_mse_ratio_vs_baseline",
            "direct_repair_policy_support_score_ratio_vs_baseline",
        ]
        present = [col for col in cols if col in selection.columns]
        lines.append("| " + " | ".join(present) + " |")
        lines.append("| " + " | ".join("---" for _ in present) + " |")
        for row in selection[present].to_dict("records"):
            vals = []
            for col in present:
                value = row.get(col)
                vals.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
            lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A non-baseline method is recommended only when every guard passes.",
            "- If strict guards fail but the relaxed fallback is enabled, the selector",
            "  can choose a candidate that improves direct repair MSE, planner-used",
            "  repair MSE, and policy-support score without exceeding final-val regret.",
            "- If neither strict nor relaxed guards pass, the selector falls back to",
            "  the same augmented-graph support+bottleneck baseline.",
            "- The selected method is an offline supervised candidate for further",
            "  validation. It is not an execution-success claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_regret_guard_outputs(
    output_dir: str | Path,
    annotated: pd.DataFrame,
    selection: pd.DataFrame,
    payload: dict[str, Any],
) -> None:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(out / "regret_guard_candidate_table.csv", index=False)
    selection.to_csv(out / "regret_guard_selection.csv", index=False)
    with (out / "regret_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (out / "regret_guard_summary.md").write_text(render_regret_guard_markdown(payload, selection), encoding="utf-8")

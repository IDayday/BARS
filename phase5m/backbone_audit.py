from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BARS_RUNS = {
    "direct_gcbc_final_goal": "direct_gcbc_3ep_corebot100k_task1_phase5l_audit",
    "bars_phase5i_state_outcome_w0p5": "state_outcome_w0p5_3ep_corebot100k_H10_B120",
    "bars_phase5k_preplan_w0p5": "preplan_policy_mismatch_w0p5_3ep_corebot100k_H10_B120",
    "bars_phase5l_progress_guard_w0p5": "edge_progress_guard_w0p5_3ep_corebot100k_H10_B120",
}


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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def load_gas_inventory(path: str | Path, env_names: list[str] | None = None) -> pd.DataFrame:
    inventory_path = Path(path).expanduser()
    inventory = pd.read_csv(inventory_path)
    if env_names:
        inventory = inventory[inventory["env_name"].astype(str).isin([str(x) for x in env_names])].copy()
    for col in [
        "keygraph_exists",
        "policy_exists",
        "tdr_exists",
        "dataset_embeddings_exists",
        "can_evaluate_official_gas",
    ]:
        if col in inventory.columns:
            inventory[col] = pd.to_numeric(inventory[col], errors="coerce").fillna(0).astype(int)
    base_dirs = [Path.cwd()]
    if inventory_path.parent.exists():
        base_dirs.append(inventory_path.parent)

    def _resolve_existing(value: Any) -> tuple[str, bool]:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "", False
        text = str(value)
        if not text:
            return "", False
        p = Path(text).expanduser()
        candidates = [p] if p.is_absolute() else [base / p for base in base_dirs]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate), True
        return str(candidates[0]), False

    for col in ["keygraph_path", "policy_path", "tdr_path", "dataset_embeddings_path"]:
        if col in inventory.columns:
            resolved = inventory[col].map(_resolve_existing)
            inventory[f"{col}_resolved"] = [x[0] for x in resolved]
            inventory[f"{col}_exists_live"] = [int(x[1]) for x in resolved]
    return inventory.reset_index(drop=True)


def load_gas_success(path: str | Path, env_names: list[str] | None = None) -> pd.DataFrame:
    success = pd.read_csv(Path(path).expanduser())
    if env_names and "env_name" in success.columns:
        success = success[success["env_name"].astype(str).isin([str(x) for x in env_names])].copy()
    return success.reset_index(drop=True)


def summarize_gas_backbones(inventory: pd.DataFrame, success: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    cols = [
        "env_name",
        "seed",
        "artifact_status",
        "can_evaluate_official_gas",
        "keygraph_path",
        "keygraph_path_resolved",
        "policy_path",
        "policy_path_resolved",
        "tdr_path",
        "tdr_path_resolved",
        "dataset_embeddings_path",
        "dataset_embeddings_path_resolved",
        "keygraph_exists",
        "keygraph_path_exists_live",
        "policy_exists",
        "policy_path_exists_live",
        "tdr_exists",
        "tdr_path_exists_live",
        "dataset_embeddings_exists",
        "dataset_embeddings_path_exists_live",
    ]
    out = inventory[[c for c in cols if c in inventory.columns]].copy()
    if not success.empty and "env_name" in success.columns:
        keep = [
            "env_name",
            "episodes",
            "success_rate",
            "success_ci95_low",
            "success_ci95_high",
            "mean_steps",
            "mean_final_goal_dist_phi",
        ]
        out = out.merge(success[[c for c in keep if c in success.columns]], on="env_name", how="left")
    out["gas_backbone_ready"] = (
        (out.get("keygraph_exists", 0).astype(int) > 0)
        & (out.get("policy_exists", 0).astype(int) > 0)
        & (out.get("tdr_exists", 0).astype(int) > 0)
        & (out.get("can_evaluate_official_gas", 0).astype(int) > 0)
    )
    live_cols = [
        "keygraph_path_exists_live",
        "policy_path_exists_live",
        "tdr_path_exists_live",
        "dataset_embeddings_path_exists_live",
    ]
    if all(col in out.columns for col in live_cols):
        out["gas_backbone_live_ready"] = np.logical_and.reduce(
            [out[col].astype(int).to_numpy() > 0 for col in live_cols]
        )
    else:
        out["gas_backbone_live_ready"] = False
    return out


def _read_episode_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_bars_runs(
    phase3f_root: str | Path,
    dataset_key: str,
    runs: dict[str, str] | None = None,
) -> pd.DataFrame:
    root = Path(phase3f_root).expanduser() / dataset_key
    rows: list[dict[str, Any]] = []
    for label, run_name in (runs or DEFAULT_BARS_RUNS).items():
        run_dir = root / run_name
        df = _read_episode_summary(run_dir / "episode_summary.csv")
        row: dict[str, Any] = {
            "run_label": label,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "exists": bool(run_dir.exists()),
            "episode_summary_exists": bool(not df.empty),
        }
        if not df.empty:
            row.update(
                {
                    "episodes": int(df.shape[0]),
                    "success_rate": float(pd.to_numeric(df["success"], errors="coerce").fillna(0.0).mean())
                    if "success" in df.columns
                    else np.nan,
                    "mean_final_goal_l2": float(pd.to_numeric(df["final_goal_l2"], errors="coerce").mean())
                    if "final_goal_l2" in df.columns
                    else np.nan,
                    "mean_l2_improvement": float(
                        (
                            pd.to_numeric(df["initial_goal_l2"], errors="coerce")
                            - pd.to_numeric(df["final_goal_l2"], errors="coerce")
                        ).mean()
                    )
                    if {"initial_goal_l2", "final_goal_l2"}.issubset(df.columns)
                    else np.nan,
                    "mean_completed_edges": float(pd.to_numeric(df["completed_edges"], errors="coerce").mean())
                    if "completed_edges" in df.columns
                    else np.nan,
                    "mean_replans": float(pd.to_numeric(df["replans"], errors="coerce").mean())
                    if "replans" in df.columns
                    else np.nan,
                    "mean_failed_edge_attempts": float(pd.to_numeric(df["failed_edge_attempts"], errors="coerce").mean())
                    if "failed_edge_attempts" in df.columns
                    else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_planner_policy_matrix(
    gas_backbones: pd.DataFrame,
    bars_runs: pd.DataFrame,
    *,
    env_name: str,
    dataset_key: str,
    phase2_run_dir: str | Path,
    bars_model_path: str | Path,
) -> pd.DataFrame:
    ready_col = "gas_backbone_live_ready" if "gas_backbone_live_ready" in gas_backbones.columns else "gas_backbone_ready"
    gas_ready = bool(not gas_backbones.empty and gas_backbones[ready_col].astype(bool).any())
    official = gas_backbones[gas_backbones["env_name"].astype(str) == str(env_name)]
    gas_success = float(official["success_rate"].dropna().iloc[0]) if "success_rate" in official and official["success_rate"].notna().any() else np.nan
    best_bars = bars_runs.copy()
    if not best_bars.empty and "success_rate" in best_bars.columns:
        best_bars = best_bars.sort_values(["success_rate", "mean_final_goal_l2"], ascending=[False, True])
    best_row = best_bars.iloc[0].to_dict() if not best_bars.empty else {}

    rows = [
        {
            "experiment_id": "official_gas_policy_official_gas_graph",
            "planner": "official_gas_keygraph_shortest_path",
            "policy": "official_gas_actor",
            "target_space": "tdr_phi_skill",
            "status": "ready_live_artifacts" if gas_ready else "blocked_missing_live_gas_artifacts",
            "primary_question": "Reference success rate under the official GAS graph, actor, and eval protocol.",
            "current_success_rate": gas_success,
            "current_mean_final_distance": np.nan,
            "required_work": "Run or reuse official evaluate_gas.py outputs.",
            "expected_value": "Strong baseline and success-protocol lock.",
        },
        {
            "experiment_id": "bars_support_graph_bars_gcbc",
            "planner": "bars_support_option_graph",
            "policy": "bars_phase3_gcbc",
            "target_space": "raw_observation_goal",
            "status": "completed_smoke" if bool(best_row) else "missing_bars_smoke",
            "primary_question": "Current BARS graph plus current BARS GCBC executor.",
            "current_success_rate": float(best_row.get("success_rate", np.nan)) if best_row else np.nan,
            "current_mean_final_distance": float(best_row.get("mean_final_goal_l2", np.nan)) if best_row else np.nan,
            "required_work": "Already available for 3-episode AntMaze smoke.",
            "expected_value": "Shows current policy/executor bottleneck; not SOTA evidence.",
        },
        {
            "experiment_id": "bars_support_graph_gas_actor",
            "planner": "bars_support_option_graph",
            "policy": "official_gas_actor",
            "target_space": "bars_raw_termination_mapped_to_tdr_phi_skill",
            "status": "diagnostic_next_if_distribution_audit_passes" if gas_ready else "blocked_missing_live_gas_artifacts",
            "primary_question": "Does support-certified BARS planning work when executed by a strong GAS low-level policy?",
            "current_success_rate": np.nan,
            "current_mean_final_distance": np.nan,
            "required_work": "First audit BARS target phi distribution against GAS keygraph/policy target distribution; only then implement GAS-phi adapter.",
            "expected_value": "Short-term diagnostic only; GAS actor is co-adapted to GAS TDR/keygraph and is not a final BARS component.",
        },
        {
            "experiment_id": "official_gas_graph_bars_gcbc",
            "planner": "official_gas_keygraph_shortest_path",
            "policy": "bars_phase3_gcbc",
            "target_space": "tdr_phi_target_to_raw_goal_mismatch",
            "status": "not_direct_without_decoder_or_nearest_raw_node",
            "primary_question": "Is the BARS policy weak even on GAS planner targets?",
            "current_success_rate": np.nan,
            "current_mean_final_distance": np.nan,
            "required_work": "Map GAS keygraph phi targets to nearest raw observations or add a raw-goal reconstruction path.",
            "expected_value": "Useful secondary isolation after BARS+GAS-policy probe.",
        },
        {
            "experiment_id": "bars_planner_subgoal_replay_policy",
            "planner": "bars_support_option_graph",
            "policy": "bars_gcbc_retrained_on_planner_subgoals",
            "target_space": "raw_observation_goal",
            "status": "next_training_path",
            "primary_question": "Can policy training on planner-issued subgoals close the execution gap?",
            "current_success_rate": np.nan,
            "current_mean_final_distance": np.nan,
            "required_work": "Build a joint graph-policy training loop: graph-derived subgoals, aligned goal/skill representation, matched sampler, and natural-start eval.",
            "expected_value": "Main algorithm path toward a complete BARS method rather than graph-only or borrowed-policy evidence.",
        },
    ]
    out = pd.DataFrame(rows)
    out["env_name"] = str(env_name)
    out["dataset_key"] = str(dataset_key)
    out["phase2_run_dir"] = str(phase2_run_dir)
    out["bars_model_path"] = str(bars_model_path)
    return out


def write_markdown_summary(
    path: str | Path,
    *,
    env_name: str,
    gas_backbones: pd.DataFrame,
    bars_runs: pd.DataFrame,
    matrix: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = int(gas_backbones["gas_backbone_ready"].sum()) if "gas_backbone_ready" in gas_backbones else 0
    live_ready = int(gas_backbones["gas_backbone_live_ready"].sum()) if "gas_backbone_live_ready" in gas_backbones else 0
    gas_success = np.nan
    if not gas_backbones.empty and "success_rate" in gas_backbones.columns and gas_backbones["success_rate"].notna().any():
        gas_success = float(gas_backbones["success_rate"].dropna().iloc[0])
    best_bars = bars_runs.copy()
    if not best_bars.empty and "success_rate" in best_bars.columns:
        best_bars = best_bars.sort_values(["success_rate", "mean_final_goal_l2"], ascending=[False, True])
    best = best_bars.iloc[0].to_dict() if not best_bars.empty else {}

    lines = [
        "# Phase 5M Policy Backbone Audit",
        "",
        "This audit separates graph/planner evidence from low-level policy evidence.",
        "The goal is to find the fastest success-rate validation path rather than",
        "continuing graph-only improvements.",
        "Cross-policy reuse is diagnostic only; final BARS evidence requires a",
        "joint graph-policy-training loop.",
        "",
        "## Evidence",
        "",
        f"- environment: `{env_name}`",
        f"- ready official GAS backbones: `{ready}`",
        f"- live local GAS backbones: `{live_ready}`",
        f"- official GAS success rate: `{gas_success:.4f}`" if np.isfinite(gas_success) else "- official GAS success rate: `nan`",
        (
            f"- best current BARS smoke: `{best.get('run_label')}` success "
            f"`{float(best.get('success_rate', np.nan)):.4f}`, mean final L2 "
            f"`{float(best.get('mean_final_goal_l2', np.nan)):.4f}`"
            if best
            else "- best current BARS smoke: `missing`"
        ),
        "",
        "## Recommended Matrix",
        "",
        "| experiment | status | policy | planner | required work |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in matrix.itertuples(index=False):
        lines.append(
            f"| `{row.experiment_id}` | `{row.status}` | `{row.policy}` | `{row.planner}` | {row.required_work} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The first diagnostic is not rollout; it is target-distribution feasibility.",
            "`bars_support_graph_gas_actor` should run only if BARS support targets mapped",
            "through GAS `get_phi` lie near the GAS keygraph/policy target distribution.",
            "The GAS actor is trained with GAS's TDR/graph/skill distribution, so raw BARS",
            "cluster/termination targets may be out of distribution even after phi mapping.",
            "",
            "The reverse composition, `official_gas_graph_bars_gcbc`, is not directly",
            "comparable because GAS planner targets live in TDR phi/skill space while the",
            "current BARS GCBC consumes raw observation goals.",
            "",
            "The main algorithm path remains `bars_planner_subgoal_replay_policy`: train",
            "the BARS low-level policy on the same graph-derived goal/skill distribution",
            "that BARS will execute at test time, then evaluate natural-start success.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_backbone_audit(
    *,
    gas_inventory_csv: str | Path,
    gas_success_csv: str | Path,
    phase3f_root: str | Path,
    dataset_key: str,
    env_name: str,
    output_dir: str | Path,
    phase2_run_dir: str | Path,
    bars_model_path: str | Path,
    bars_runs: dict[str, str] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    inventory = load_gas_inventory(gas_inventory_csv, env_names=[env_name])
    success = load_gas_success(gas_success_csv, env_names=[env_name])
    gas_backbones = summarize_gas_backbones(inventory, success)
    bars_summary = summarize_bars_runs(phase3f_root, dataset_key, runs=bars_runs)
    matrix = build_planner_policy_matrix(
        gas_backbones,
        bars_summary,
        env_name=env_name,
        dataset_key=dataset_key,
        phase2_run_dir=phase2_run_dir,
        bars_model_path=bars_model_path,
    )
    gas_backbones.to_csv(out / "gas_backbone_inventory.csv", index=False)
    bars_summary.to_csv(out / "bars_policy_smoke_summary.csv", index=False)
    matrix.to_csv(out / "planner_policy_matrix.csv", index=False)
    summary = {
        "env_name": env_name,
        "dataset_key": dataset_key,
        "num_ready_gas_backbones": int(gas_backbones["gas_backbone_ready"].sum()) if "gas_backbone_ready" in gas_backbones else 0,
        "num_live_ready_gas_backbones": int(gas_backbones["gas_backbone_live_ready"].sum())
        if "gas_backbone_live_ready" in gas_backbones
        else 0,
        "official_gas_success_rate": float(gas_backbones["success_rate"].dropna().iloc[0])
        if "success_rate" in gas_backbones and gas_backbones["success_rate"].notna().any()
        else None,
        "recommended_next_experiment": "bars_vs_gas_target_distribution_audit",
        "note": (
            "Cross-policy reuse is diagnostic only. Final BARS evidence requires "
            "joint graph construction, policy training, goal/skill representation, "
            "and execution under one algorithm."
        ),
    }
    _write_json(out / "phase5m_policy_backbone_audit_summary.json", summary)
    write_markdown_summary(
        out / "phase5m_policy_backbone_audit_summary.md",
        env_name=env_name,
        gas_backbones=gas_backbones,
        bars_runs=bars_summary,
        matrix=matrix,
    )
    return {
        "gas_backbones": gas_backbones,
        "bars_runs": bars_summary,
        "matrix": matrix,
        "summary": summary,
        "output_dir": out,
    }

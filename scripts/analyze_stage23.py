#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    if len(df) == 0 and len(df.columns) == 0:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def _status_from_metric(df: pd.DataFrame, col: str, threshold: float, pass_name: str, hold_name: str) -> str:
    if len(df) == 0 or col not in df:
        return "PENDING"
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) and vals.max() >= threshold:
        return pass_name
    return hold_name


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if len(df) == 0 or col not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _repro_adapter_gaps(repro: pd.DataFrame) -> pd.DataFrame:
    if len(repro) == 0 or "success" not in repro:
        return pd.DataFrame()
    rows = []
    for (env, seed), sub in repro.groupby(["env", "seed"], dropna=False):
        def route_success(route: str) -> float | None:
            r = sub[sub["route"].astype(str).eq(route)]
            if len(r) == 0:
                return None
            vals = pd.to_numeric(r["success"], errors="coerce").dropna()
            return float(vals.iloc[-1]) if len(vals) else None

        b = route_success("B_official_our_checkpoint")
        c = route_success("C_adapter_same_checkpoint")
        if b is not None and c is not None:
            rows.append({"env": env, "seed": seed, "official_B_success": b, "adapter_C_success": c, "adapter_minus_official_pp": 100.0 * (c - b)})
    return pd.DataFrame(rows)


def _oracle_decision(oracle: pd.DataFrame) -> str:
    if len(oracle) == 0:
        return "PENDING_ORACLE"
    go = False
    for _, row in oracle[oracle.get("graph_id", "").astype(str).eq("G_oracle")].iterrows():
        reduction = float(pd.to_numeric(pd.Series([row.get("mean_path_cost_reduction", 0)]), errors="coerce").fillna(0).iloc[0])
        shorter = float(pd.to_numeric(pd.Series([row.get("shorter_path_rate", 0)]), errors="coerce").fillna(0).iloc[0])
        usage = float(pd.to_numeric(pd.Series([row.get("bridge_usage_rate", 0)]), errors="coerce").fillna(0).iloc[0])
        if reduction >= 0.5 or (shorter >= 0.2 and usage >= 0.2):
            go = True
    return "PASS_ORACLE" if go else "NO_ORACLE_UPPER_BOUND"


def _pbridge_decision(pbridge: pd.DataFrame) -> str:
    if len(pbridge) == 0:
        return "PENDING_P_BRIDGE"
    auroc = _num_series(pbridge, "selected_bridge_AUROC").max()
    fp = _num_series(pbridge, "false_positive_bridge_relative_reduction@0.6").max()
    if pd.notna(auroc) and auroc >= 0.65 and pd.notna(fp) and fp >= 0.20:
        return "PASS_P_BRIDGE"
    if pd.notna(auroc) and auroc >= 0.65:
        return "PARTIAL_P_BRIDGE_HOLD_FP_REDUCTION"
    return "HOLD_P_BRIDGE"


def _boundary_decision(boundary: pd.DataFrame) -> str:
    if len(boundary) == 0:
        return "PENDING_BOUNDARY"
    auroc = _num_series(boundary, "psi_AUROC_for_conditional_success").max()
    gap = _num_series(boundary, "supported_gap").max()
    coverage = _num_series(boundary, "coverage").max()
    if pd.notna(auroc) and auroc >= 0.60 and pd.notna(gap) and gap >= 0.10 and pd.notna(coverage) and coverage > 0.01:
        return "PASS_BOUNDARY"
    return "HOLD_BOUNDARY_DIAGNOSTIC_ONLY"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--runs-root", default=".")
    p.add_argument("--reports-root", default="reports")
    p.add_argument("--out", default="reports/stage23_summary.md")
    # Backward-compatible aliases from older Stage23 script.
    p.add_argument("--eval-root", default="")
    p.add_argument("--artifact-root-stage22", default="")
    args = p.parse_args()
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)

    repro = _read(reports / "stage23_gas_reproduction_matrix.csv")
    repair = _read(reports / "stage23_adapter_protocol_repair.csv")
    protocol = _read(reports / "stage23_protocol_audit.csv")
    atlas = _read(reports / "stage23_failure_atlas_grouped.csv")
    bridge = _read(reports / "stage23_bridge_graph_summary.csv")
    edge_exec = _read(reports / "stage23_edge_execution_summary.csv")
    oracle = _read(reports / "stage23_oracle_bridge_summary.csv")
    pbridge = _read(reports / "stage23_p_bridge_metrics.csv")
    boundary = _read(reports / "stage23_boundary_junction_metrics.csv")
    integrated = _collect_integrated(Path(args.runs_root))
    _write_csv(integrated, reports / "stage23_integrated_results.csv", ["env", "seed", "variant", "episodes", "success", "steps", "eval_path"])
    if len(integrated):
        comp = integrated.groupby([c for c in ["env", "seed", "variant"] if c in integrated.columns], dropna=False).agg(episodes=("success", "count"), success=("success", "mean"), steps=("steps", "mean")).reset_index()
    else:
        comp = pd.DataFrame()
    _write_csv(comp, reports / "stage23_integrated_comparison.csv", ["env", "seed", "variant", "episodes", "success", "steps"])
    fallback = _read(reports / "stage23_fallback_causal.csv")
    if len(fallback) == 0:
        _write_csv(fallback, reports / "stage23_fallback_causal.csv", ["env", "seed", "task_id", "episode_id", "trigger_id", "condition", "success", "steps"])

    lines = ["# Stage23 Summary", ""]
    lines.append("## 1. Official GAS Reproduction")
    repro_gaps = _repro_adapter_gaps(repro)
    if len(repro):
        completed = repro[repro["status"].astype(str).eq("completed")]
        skipped = repro[repro["status"].astype(str).eq("skipped")]
        failed = repro[repro["status"].astype(str).eq("failed")]
        lines.append(f"- Matrix rows: {len(repro)}; completed {len(completed)}, skipped {len(skipped)}, failed {len(failed)}.")
        if len(repro_gaps):
            try:
                lines.append(repro_gaps.to_markdown(index=False))
            except Exception:
                lines.append("```csv\n" + repro_gaps.to_csv(index=False).strip() + "\n```")
            worst_gap = float(repro_gaps["adapter_minus_official_pp"].abs().max())
            lines.append(f"- Adapter-vs-official max absolute gap: {worst_gap:.1f}pp.")
        if len(skipped):
            lines.append("- Raw three-route matrix has skipped official-pretrained rows; the repaired control route is used for the current gate.")
        if len(repro_gaps) and float(repro_gaps["adapter_minus_official_pp"].abs().max()) > 3.0:
            lines.append("- Original route C differs from route B by more than 3pp, so adapter conclusions use the repaired official-control path.")
        if len(repair):
            keep = [c for c in ["env", "seed", "official_B_success", "adapter_original_success", "adapter_official_control_success", "official_control_minus_official_pp", "episodes", "mean_steps"] if c in repair]
            lines.append("- Protocol repair route:")
            try:
                lines.append(repair[keep].to_markdown(index=False))
            except Exception:
                lines.append("```csv\n" + repair[keep].to_csv(index=False).strip() + "\n```")
            repair_gap = _num_series(repair, "official_control_minus_official_pp").abs().max()
            if pd.notna(repair_gap) and repair_gap <= 3.0:
                lines.append(f"- Repaired official-control adapter is within {repair_gap:.1f}pp of official route B.")
                lines.append("- Reproduction gate: GO_REPRO_REPAIRED.")
    else:
        lines.append("- PENDING: reproduction matrix not available.")
    if len(protocol):
        bad = protocol[protocol.get("protocol_status", "").astype(str) != "ok"] if "protocol_status" in protocol else pd.DataFrame()
        lines.append(f"- Protocol audit rows: {len(protocol)}; non-ok rows: {len(bad)}.")
    lines.append("")

    lines.append("## 2. Failure Atlas")
    if len(atlas):
        try:
            lines.append(atlas.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + atlas.to_csv(index=False).strip() + "\n```")
        failures = atlas[atlas.get("primary_failure_type", "") != "SUCCESS"] if "primary_failure_type" in atlas else atlas
        if len(failures):
            top = failures.sort_values("episodes", ascending=False).iloc[0]
            lines.append(f"- Main failure mode: {top.get('primary_failure_type')} ({int(top.get('episodes', 0))} grouped episodes).")
    else:
        lines.append("- PENDING: no failure atlas rows.")
    lines.append("")

    lines.append("## 3. Bridge And Oracle")
    if len(bridge):
        headroom = bridge[(bridge.get("graph_id", "") != "G0") & ((pd.to_numeric(bridge.get("shorter_path_rate", 0), errors="coerce").fillna(0) > 0) | (pd.to_numeric(bridge.get("bridge_usage_rate", 0), errors="coerce").fillna(0) > 0))]
        lines.append(f"- Bridge existence: {'PASS_BRIDGE_EXISTENCE' if len(headroom) else 'NO_BRIDGE_HEADROOM'} ({len(headroom)} graph rows with shorter/bridge-using paths).")
    else:
        lines.append("- Bridge existence: PENDING.")
    if len(edge_exec):
        weak = "set_state_rate" in edge_exec and pd.to_numeric(edge_exec["set_state_rate"], errors="coerce").fillna(0).max() < 0.5
        lines.append(f"- Edge execution labels: {'weak proxy' if weak else 'rollout-backed'}; rows by type available.")
        try:
            lines.append(edge_exec.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + edge_exec.to_csv(index=False).strip() + "\n```")
    else:
        lines.append("- Edge execution: PENDING.")
    if len(oracle):
        lines.append(f"- Oracle gate: {_oracle_decision(oracle)}.")
        try:
            lines.append(oracle.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + oracle.to_csv(index=False).strip() + "\n```")
    else:
        lines.append("- Oracle bridge: PENDING.")
    lines.append("")

    lines.append("## 4. p_bridge And Boundary")
    lines.append(f"- p_bridge gate: {_pbridge_decision(pbridge)}.")
    if len(pbridge):
        keep_cols = [c for c in ["env", "seed", "selected_bridge_AUROC", "selected_bridge_AUPRC", "selected_bridge_base_success_rate", "accepted_bridge_success_rate@0.6", "accepted_bridge_success_rate@0.7", "false_positive_bridge_relative_reduction@0.6"] if c in pbridge]
        try:
            lines.append(pbridge[keep_cols].to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + pbridge[keep_cols].to_csv(index=False).strip() + "\n```")
    lines.append(f"- Boundary gate: {_boundary_decision(boundary)}.")
    if len(boundary):
        try:
            lines.append(boundary.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + boundary.to_csv(index=False).strip() + "\n```")
    lines.append("")

    lines.append("## 5. Integrated BARS-v3")
    if len(comp):
        try:
            lines.append(comp.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + comp.to_csv(index=False).strip() + "\n```")
        lines.append("- Integrated decision needs paired comparison against aggressive and oracle variants before GO_BARS_V3.")
    else:
        lines.append("- PENDING: no integrated no-fallback eval rows.")
    lines.append("")

    lines.append("## 6. Fallback Causal")
    if len(fallback):
        lines.append("- Fallback causal ablation rows exist.")
    else:
        lines.append("- HOLD_FALLBACK: causal trigger-state ablation has not been run.")
    lines.append("")

    decision = "HOLD_REPRO"
    repro_ok = False
    if len(repro):
        no_failed = len(repro[repro["status"].astype(str).eq("failed")]) == 0
        gap_ok = len(repro_gaps) > 0 and float(repro_gaps["adapter_minus_official_pp"].abs().max()) <= 3.0
        repair_gap = _num_series(repair, "official_control_minus_official_pp").abs().max() if len(repair) else float("nan")
        repair_ok = pd.notna(repair_gap) and repair_gap <= 3.0
        repro_ok = no_failed and (gap_ok or repair_ok)
    if repro_ok:
        decision = "GO_REPRO"
    if _oracle_decision(oracle) == "NO_ORACLE_UPPER_BOUND":
        decision = "NO_BARS_HEADROOM_ON_TESTED_ORACLE_ENV"
    if _pbridge_decision(pbridge).startswith("HOLD"):
        decision = "HOLD_P_BRIDGE"
    if not repro_ok:
        decision = "HOLD_REPRO"
    lines.append("## 7. Current Decision")
    lines.append(f"- {decision}")
    lines.append("")
    lines.append("## 8. Next Commands")
    lines.append("```bash")
    if decision == "NO_BARS_HEADROOM_ON_TESTED_ORACLE_ENV":
        lines.append("# Do not run integrated BARS-v3 on the tested antmaze hard envs until a new oracle upper bound appears.")
        lines.append("PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage22_prepare_gas_backbone.sh ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} ARTIFACT_ROOT=artifacts/gas PREFER_PRETRAINED=1 TRAIN_IF_MISSING=0 LOG_ROOT=runs_stage23_prepare_scene")
        lines.append("PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage23_pipeline.sh MODE=bridge ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1")
        lines.append("PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage23_pipeline.sh MODE=edge_exec ENVS=scene-play-v0 SEEDS=0 EDGE_EXEC_PILOT=1 GPUS=${GPUS:-0} WAIT=1")
        lines.append("PATH=/root/anaconda3/envs/gcrlo/bin:$PATH PYTHONPATH=$PWD bash scripts/stage23_pipeline.sh MODE=oracle ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1")
    else:
        lines.append("bash scripts/stage23_pipeline.sh MODE=repro ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1")
        lines.append("bash scripts/stage23_pipeline.sh MODE=bridge ENVS=antmaze-large-explore-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1")
        lines.append("bash scripts/stage23_pipeline.sh MODE=edge_exec ENVS=antmaze-large-explore-v0 SEEDS=0 EDGE_EXEC_PILOT=1 GPUS=${GPUS:-0} WAIT=1")
    lines.append("```")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")


def _collect_integrated(runs_root: Path) -> pd.DataFrame:
    frames = []
    for path in runs_root.rglob("runs_stage23_integrated/**/eval.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["eval_path"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    main()

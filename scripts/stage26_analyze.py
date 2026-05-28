from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import platform
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _read_eval_args(eval_path: Path) -> dict:
    args_path = eval_path.parent / "eval_args.json"
    if not args_path.exists():
        return {}
    try:
        return json.load(open(args_path, encoding="utf-8"))
    except Exception:
        return {}


def _variant_from_args(eval_path: Path, row: dict, eval_args: dict) -> tuple[str, str]:
    mode = row.get("mode", "") or str(eval_args.get("mode", ""))
    if mode == "gas_graph_policy":
        return "gas", ""
    if mode == "gas_graph_lowcond_policy":
        eval_variant = str(eval_args.get("variant") or row.get("variant") or "unknown")
        actor_parent = Path(str(eval_args.get("actor", ""))).parent.name
        if actor_parent:
            train_label = actor_parent
            for suffix in ("_bc_50k_s5k", "_bc50k_s5k"):
                if train_label.endswith(suffix):
                    train_label = train_label[: -len(suffix)]
            if eval_variant == "full_localres" and not train_label.startswith("full_localres"):
                train_label = f"full_localres_from_{train_label}"
            elif eval_variant != "full" and not train_label.startswith(eval_variant):
                train_label = f"{eval_variant}_from_{train_label}"
            return f"lowcond_{train_label}", ""
        return f"lowcond_{eval_variant}", ""
    weight = ""
    if eval_args:
        weight = str(eval_args.get("tmd_cost_weight", ""))
    if not weight:
        name = eval_path.parent.name
        if "tmdcost_w" in name:
            weight = name.split("tmdcost_w", 1)[1].split("_", 1)[0]
    return "tmd_cost", weight


def _phase_from_path(eval_path: Path) -> str:
    parts = eval_path.parts
    for part in parts:
        if part.startswith("phase_"):
            return part
    return ""


def _collect_eval_rows(runs_root: str) -> list[dict]:
    rows: list[dict] = []
    for path_text in glob.glob(f"{runs_root}/**/eval.csv", recursive=True):
        path = Path(path_text)
        run_rows = _read_csv(path)
        eval_args = _read_eval_args(path)
        variant, weight = _variant_from_args(path, run_rows[0] if run_rows else {}, eval_args)
        for row in run_rows:
            row["run_name"] = path.parent.name
            row["eval_path"] = str(path)
            row["phase"] = _phase_from_path(path)
            row["variant"] = variant
            row["tmd_cost_weight"] = weight
            row["run_episodes"] = str(eval_args.get("episodes", ""))
        rows.extend(run_rows)
    return rows


def _collect_graph_rows(artifacts_root: str) -> list[dict]:
    rows: list[dict] = []
    for path_text in glob.glob(f"{artifacts_root}/**/graph_stats.json", recursive=True):
        path = Path(path_text)
        try:
            row = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        row["path"] = str(path)
        rows.append(row)
    return rows


def _group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return out


def _summary_row(rows: list[dict]) -> dict:
    n = len(rows)
    success = sum(_safe_float(r.get("success")) for r in rows)
    steps = sum(_safe_float(r.get("steps")) for r in rows) / max(n, 1)
    no_path = sum(_safe_float(r.get("no_path_rate")) for r in rows) / max(n, 1)
    final_steps = sum(_safe_float(r.get("final_goal_mode_steps")) for r in rows) / max(n, 1)
    switches = sum(_safe_float(r.get("subgoal_switch_count")) for r in rows) / max(n, 1)
    return {
        "episodes": n,
        "success": success,
        "success_rate": success / max(n, 1),
        "mean_steps": steps,
        "mean_no_path_rate": no_path,
        "mean_final_goal_steps": final_steps,
        "mean_subgoal_switches": switches,
    }


def _normal_ci_delta(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    se = math.sqrt(max(p1 * (1.0 - p1), 0.0) / max(n1, 1) + max(p2 * (1.0 - p2), 0.0) / max(n2, 1))
    delta = p2 - p1
    return delta - 1.96 * se, delta + 1.96 * se


def _bootstrap_delta(gas_rows: list[dict], var_rows: list[dict], samples: int = 2000) -> tuple[float, float] | tuple[str, str]:
    gas_by = _group(gas_rows, ("seed", "gas_seed", "task_id"))
    var_by = _group(var_rows, ("seed", "gas_seed", "task_id"))
    strata = sorted(set(gas_by) & set(var_by))
    if not strata:
        return "", ""
    rng = random.Random(2601)
    deltas: list[float] = []
    for _ in range(samples):
        gas_success = gas_n = var_success = var_n = 0.0
        sampled = [rng.choice(strata) for _ in strata]
        for key in sampled:
            gr = gas_by[key]
            vr = var_by[key]
            gas_success += sum(_safe_float(r.get("success")) for r in gr)
            gas_n += len(gr)
            var_success += sum(_safe_float(r.get("success")) for r in vr)
            var_n += len(vr)
        deltas.append(var_success / max(var_n, 1) - gas_success / max(gas_n, 1))
    deltas.sort()
    lo = deltas[int(0.025 * (samples - 1))]
    hi = deltas[int(0.975 * (samples - 1))]
    return lo, hi


def _phase_b_comparisons(eval_rows: list[dict]) -> list[dict]:
    by_env = _group(eval_rows, ("env", "run_episodes"))
    comparisons: list[dict] = []
    for (env_name, run_episodes), env_rows in sorted(by_env.items()):
        gas_rows = [r for r in env_rows if r.get("variant") == "gas"]
        for weight in sorted({r.get("tmd_cost_weight", "") for r in env_rows if r.get("variant") == "tmd_cost"}):
            if not weight:
                continue
            var_rows = [r for r in env_rows if r.get("variant") == "tmd_cost" and r.get("tmd_cost_weight") == weight]
            if not gas_rows or not var_rows:
                continue
            g = _summary_row(gas_rows)
            v = _summary_row(var_rows)
            lo, hi = _normal_ci_delta(g["success_rate"], g["episodes"], v["success_rate"], v["episodes"])
            blo, bhi = _bootstrap_delta(gas_rows, var_rows)
            comparisons.append(
                {
                    "env": env_name,
                    "run_episodes": run_episodes,
                    "weight": weight,
                    "gas_n": g["episodes"],
                    "gas_success": g["success"],
                    "gas_success_rate": g["success_rate"],
                    "variant_n": v["episodes"],
                    "variant_success": v["success"],
                    "variant_success_rate": v["success_rate"],
                    "delta_success_rate": v["success_rate"] - g["success_rate"],
                    "normal95_low": lo,
                    "normal95_high": hi,
                    "bootstrap95_low": blo,
                    "bootstrap95_high": bhi,
                    "delta_mean_steps": v["mean_steps"] - g["mean_steps"],
                    "delta_no_path_rate": v["mean_no_path_rate"] - g["mean_no_path_rate"],
                    "delta_final_goal_steps": v["mean_final_goal_steps"] - g["mean_final_goal_steps"],
                    "delta_subgoal_switches": v["mean_subgoal_switches"] - g["mean_subgoal_switches"],
                }
            )
    return comparisons


def _task_comparisons(eval_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    by = _group(eval_rows, ("env", "run_episodes", "task_id"))
    for (env_name, run_episodes, task_id), task_rows in sorted(by.items()):
        gas_rows = [r for r in task_rows if r.get("variant") == "gas"]
        if not gas_rows:
            continue
        g = _summary_row(gas_rows)
        for weight in sorted({r.get("tmd_cost_weight", "") for r in task_rows if r.get("variant") == "tmd_cost"}):
            var_rows = [r for r in task_rows if r.get("variant") == "tmd_cost" and r.get("tmd_cost_weight") == weight]
            if not var_rows:
                continue
            v = _summary_row(var_rows)
            rows.append(
                {
                    "env": env_name,
                    "run_episodes": run_episodes,
                    "task_id": task_id,
                    "weight": weight,
                    "gas_n": g["episodes"],
                    "gas_success_rate": g["success_rate"],
                    "variant_n": v["episodes"],
                    "variant_success_rate": v["success_rate"],
                    "delta_success_rate": v["success_rate"] - g["success_rate"],
                    "delta_mean_steps": v["mean_steps"] - g["mean_steps"],
                }
            )
    return rows


def _lowcond_comparisons(eval_rows: list[dict]) -> list[dict]:
    by_env = _group(eval_rows, ("env", "run_episodes"))
    comparisons: list[dict] = []
    for (env_name, run_episodes), env_rows in sorted(by_env.items()):
        gas_rows = [r for r in env_rows if r.get("variant") == "gas"]
        for variant in sorted({r.get("variant", "") for r in env_rows if str(r.get("variant", "")).startswith("lowcond_")}):
            var_rows = [r for r in env_rows if r.get("variant") == variant]
            if not gas_rows or not var_rows:
                continue
            g = _summary_row(gas_rows)
            v = _summary_row(var_rows)
            lo, hi = _normal_ci_delta(g["success_rate"], g["episodes"], v["success_rate"], v["episodes"])
            blo, bhi = _bootstrap_delta(gas_rows, var_rows)
            comparisons.append(
                {
                    "env": env_name,
                    "run_episodes": run_episodes,
                    "variant": variant,
                    "gas_n": g["episodes"],
                    "gas_success": g["success"],
                    "gas_success_rate": g["success_rate"],
                    "variant_n": v["episodes"],
                    "variant_success": v["success"],
                    "variant_success_rate": v["success_rate"],
                    "delta_success_rate": v["success_rate"] - g["success_rate"],
                    "normal95_low": lo,
                    "normal95_high": hi,
                    "bootstrap95_low": blo,
                    "bootstrap95_high": bhi,
                    "delta_mean_steps": v["mean_steps"] - g["mean_steps"],
                    "delta_final_goal_steps": v["mean_final_goal_steps"] - g["mean_final_goal_steps"],
                    "delta_subgoal_switches": v["mean_subgoal_switches"] - g["mean_subgoal_switches"],
                }
            )
    return comparisons


def _lowcond_task_comparisons(eval_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    by = _group(eval_rows, ("env", "run_episodes", "task_id"))
    for (env_name, run_episodes, task_id), task_rows in sorted(by.items()):
        gas_rows = [r for r in task_rows if r.get("variant") == "gas"]
        if not gas_rows:
            continue
        g = _summary_row(gas_rows)
        for variant in sorted({r.get("variant", "") for r in task_rows if str(r.get("variant", "")).startswith("lowcond_")}):
            var_rows = [r for r in task_rows if r.get("variant") == variant]
            if not var_rows:
                continue
            v = _summary_row(var_rows)
            rows.append(
                {
                    "env": env_name,
                    "run_episodes": run_episodes,
                    "task_id": task_id,
                    "variant": variant,
                    "gas_n": g["episodes"],
                    "gas_success_rate": g["success_rate"],
                    "variant_n": v["episodes"],
                    "variant_success_rate": v["success_rate"],
                    "delta_success_rate": v["success_rate"] - g["success_rate"],
                    "delta_mean_steps": v["mean_steps"] - g["mean_steps"],
                }
            )
    return rows


def _failure_rows(eval_rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, int] = defaultdict(int)
    for row in eval_rows:
        if _safe_float(row.get("success")) >= 0.5:
            continue
        if _safe_float(row.get("no_path_rate")) > 0.05:
            bucket = "no_path_or_disconnected_graph"
        elif _safe_float(row.get("final_goal_mode_steps")) > 0.5 * max(_safe_float(row.get("steps")), 1.0):
            bucket = "final_goal_handoff_failure"
        elif _safe_float(row.get("subgoal_switch_count")) > 20:
            bucket = "subgoal_unreachable_by_low_actor"
        else:
            bucket = "unclassified_execution_failure"
        key = (row.get("env", ""), row.get("variant", ""), row.get("tmd_cost_weight", ""), bucket)
        buckets[key] += 1
    return [
        {"env": k[0], "variant": k[1], "weight": k[2], "failure_type": k[3], "count": v}
        for k, v in sorted(buckets.items())
    ]


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _markdown_table(rows: list[dict], keys: list[str]) -> str:
    if not rows:
        return "No rows.\n"
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def _git_info() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception as exc:
            return f"ERR:{exc}"
    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": run(["git", "status", "--short"]),
    }


def _write_reports(args, eval_rows: list[dict], graph_rows: list[dict]) -> None:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    comparisons = _phase_b_comparisons(eval_rows)
    task_rows = _task_comparisons(eval_rows)
    lowcond_comparisons = _lowcond_comparisons(eval_rows)
    lowcond_task_rows = _lowcond_task_comparisons(eval_rows)
    failures = _failure_rows(eval_rows)
    _write_csv(reports / "stage26_tmd_tdr_eval_all.csv", eval_rows)
    _write_csv(reports / "stage26_tmd_tdr_graph_all.csv", graph_rows)
    _write_csv(reports / "stage26_tmd_tdr_phase_b_comparisons.csv", comparisons)
    _write_csv(reports / "stage26_tmd_tdr_phase_b_taskwise.csv", task_rows)
    _write_csv(reports / "stage26_tmd_tdr_lowcond_comparisons.csv", lowcond_comparisons)
    _write_csv(reports / "stage26_tmd_tdr_lowcond_taskwise.csv", lowcond_task_rows)
    _write_csv(reports / "stage26_tmd_tdr_failure_counts.csv", failures)
    git = _git_info()
    manifest = Path(args.runs_root) / "stage26_phase_b_manifest.tsv"
    all_manifest = Path(args.runs_root) / "stage26_phase_b_manifest_all.tsv"
    manifest_source = all_manifest if all_manifest.exists() else manifest
    manifest_text = manifest_source.read_text(encoding="utf-8") if manifest_source.exists() else "No run manifest yet.\n"
    protocol = f"""# Stage 26 TMD/TDR Protocol

Branch: `{git['branch']}`
Commit: `{git['commit']}`
Dataset root: `{args.dataset_root}`
Runs root: `{args.runs_root}`

## Current Scope

This report tracks Stage 26 progress against `tmd_gas_tdr_experiment_plan_and_codex_goal.md`.
The first active matrix is Phase B: seed-matched GAS graph baseline versus GAS graph plus TMD soft-cost blend.

## Reliability Rules

- Every launched run is recorded in `runs_stage26_tmd_tdr/stage26_phase_b_manifest.tsv`.
- Variants are compared only against GAS rows with matching env, env seed, GAS seed, task set, and episode budget when those rows are available.
- Pilot rows guide follow-up selection; confirmed claims require the pre-registered CI and task-wise gates.
- `fallback=none` is preserved by the underlying evaluator.
"""
    (reports / "stage26_tmd_tdr_protocol.md").write_text(protocol.rstrip() + "\n", encoding="utf-8")
    manifest_report = f"""# Stage 26 TMD/TDR Manifest

## Environment

- Branch: `{git['branch']}`
- Commit: `{git['commit']}`
- Python platform: `{platform.platform()}`
- Dataset root: `{args.dataset_root}`
- Runs root: `{args.runs_root}`

## Dirty State

```text
{git['dirty']}
```

## Phase B Run Manifest

Source: `{manifest_source}`

```tsv
{manifest_text}
```
"""
    (reports / "stage26_tmd_tdr_manifest.md").write_text(manifest_report.rstrip() + "\n", encoding="utf-8")
    summary = f"""# Stage 26 TMD/TDR Summary

Evaluation rows: {len(eval_rows)}
Graph rows: {len(graph_rows)}

## Phase B Aggregate Comparisons

{_markdown_table(comparisons, ['env', 'run_episodes', 'weight', 'gas_n', 'gas_success_rate', 'variant_n', 'variant_success_rate', 'delta_success_rate', 'normal95_low', 'normal95_high', 'bootstrap95_low', 'bootstrap95_high', 'delta_mean_steps'])}

## Phase B Task-Wise Deltas

{_markdown_table(task_rows, ['env', 'run_episodes', 'task_id', 'weight', 'gas_success_rate', 'variant_success_rate', 'delta_success_rate', 'delta_mean_steps'])}

## Phase D Low-Level Condition Comparisons

{_markdown_table(lowcond_comparisons, ['env', 'run_episodes', 'variant', 'gas_n', 'gas_success_rate', 'variant_n', 'variant_success_rate', 'delta_success_rate', 'normal95_low', 'normal95_high', 'bootstrap95_low', 'bootstrap95_high', 'delta_mean_steps'])}

## Phase D Low-Level Condition Task-Wise Deltas

{_markdown_table(lowcond_task_rows, ['env', 'run_episodes', 'task_id', 'variant', 'gas_success_rate', 'variant_success_rate', 'delta_success_rate', 'delta_mean_steps'])}
"""
    (reports / "stage26_tmd_tdr_summary.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
    failure_report = f"""# Stage 26 TMD/TDR Failure Analysis

The current taxonomy is based on evaluator-level signals available in `eval.csv`.
More detailed trigger-state replay remains required for Phase F.

{_markdown_table(failures, ['env', 'variant', 'weight', 'failure_type', 'count'])}
"""
    (reports / "stage26_tmd_tdr_failure_analysis.md").write_text(failure_report.rstrip() + "\n", encoding="utf-8")
    decision_lines = ["# Stage 26 TMD/TDR Decisions", ""]
    for row in comparisons:
        low = _safe_float(row.get("bootstrap95_low"), _safe_float(row.get("normal95_low")))
        delta = _safe_float(row.get("delta_success_rate"))
        if delta >= 0.03 and low > 0:
            decision = "PROMOTE_FOR_CONFIRM"
        elif delta > 0:
            decision = "EXPLORATORY_POSITIVE"
        else:
            decision = "DO_NOT_PROMOTE"
        decision_lines.append(
            f"- `{row['env']}` episodes `{row.get('run_episodes', '')}` weight `{row['weight']}`: {decision}; "
            f"delta={delta:.3f}, bootstrap95=[{row.get('bootstrap95_low')}, {row.get('bootstrap95_high')}], "
            f"normal95=[{_fmt(row.get('normal95_low'))}, {_fmt(row.get('normal95_high'))}]."
        )
    if lowcond_comparisons:
        decision_lines.extend(["", "## Phase D Low-Level Condition"])
    for row in lowcond_comparisons:
        low = _safe_float(row.get("bootstrap95_low"), _safe_float(row.get("normal95_low")))
        delta = _safe_float(row.get("delta_success_rate"))
        if delta >= 0.0 and low >= -0.02 and _safe_float(row.get("delta_mean_steps")) <= 0:
            decision = "PROMISING_EXECUTION"
        elif delta > -0.03:
            decision = "MIXED_OR_UNDERPOWERED"
        else:
            decision = "DO_NOT_PROMOTE"
        decision_lines.append(
            f"- `{row['env']}` episodes `{row.get('run_episodes', '')}` variant `{row['variant']}`: {decision}; "
            f"delta={delta:.3f}, bootstrap95=[{row.get('bootstrap95_low')}, {row.get('bootstrap95_high')}], "
            f"normal95=[{_fmt(row.get('normal95_low'))}, {_fmt(row.get('normal95_high'))}], "
            f"delta_steps={_fmt(row.get('delta_mean_steps'))}."
        )
    if not comparisons:
        decision_lines.append("- No complete Phase B comparisons yet.")
    (reports / "stage26_tmd_tdr_decisions.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage-26 TMD/TDR outputs.")
    parser.add_argument("--runs-root", default="runs_stage26_tmd_tdr")
    parser.add_argument("--artifacts-root", default="artifacts/tmd_test")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dataset-root", default="/mnt/project/offlinerl_datasets/ogbench")
    args = parser.parse_args(argv)
    eval_rows = _collect_eval_rows(args.runs_root)
    graph_rows = _collect_graph_rows(args.artifacts_root)
    _write_reports(args, eval_rows, graph_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

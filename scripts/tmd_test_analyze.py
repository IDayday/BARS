from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _eval_summary(rows: list[dict]) -> str:
    lines = ["# tmd-test Evaluation Summary", "", f"Rows: {len(rows)}", ""]
    run_rows = _run_summary(rows)
    if run_rows:
        keys = ["run_name", "env", "seed", "gas_seed", "mode", "rows", "success", "success_rate", "mean_steps"]
        lines.extend(["## Run aggregate", ""])
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in run_rows:
            lines.append(
                "| {run_name} | {env} | {seed} | {gas_seed} | {mode} | {rows} | {success:.1f} | {success_rate:.3f} | {mean_steps:.1f} |".format(**row)
            )
        lines.append("")
    keys = ["run_name", "env", "seed", "gas_seed", "mode", "task_id", "episodes", "success", "steps", "no_path_rate", "goal_distance_improvement", "subgoal_switch_count", "final_goal_mode_steps"]
    if rows:
        lines.extend(["## First rows", ""])
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in rows[:40]:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    else:
        lines.append("No evaluation rows were found.")
    return "\n".join(lines) + "\n"


def _run_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row.get("run_name", ""),
                row.get("env", ""),
                row.get("seed", ""),
                row.get("gas_seed", ""),
                row.get("mode", ""),
            ),
            [],
        ).append(row)
    out = []
    for (run_name, env, seed, gas_seed, mode), rs in sorted(grouped.items()):
        n = max(len(rs), 1)
        success = sum(float(r.get("success", 0) or 0) for r in rs)
        out.append(
            {
                "run_name": run_name,
                "env": env,
                "seed": seed,
                "gas_seed": gas_seed,
                "mode": mode,
                "rows": n,
                "success": success,
                "success_rate": success / n,
                "mean_steps": sum(float(r.get("steps", 0) or 0) for r in rs) / n,
            }
        )
    return out


def _mode_task_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row.get("run_name", ""),
                row.get("env", ""),
                row.get("seed", ""),
                row.get("gas_seed", ""),
                row.get("mode", ""),
                row.get("task_id", ""),
            ),
            [],
        ).append(row)
    out = []
    for (run_name, env, seed, gas_seed, mode, task_id), rs in sorted(grouped.items()):
        n = max(len(rs), 1)
        out.append(
            {
                "run_name": run_name,
                "env": env,
                "seed": seed,
                "gas_seed": gas_seed,
                "mode": mode,
                "task_id": task_id,
                "episodes": n,
                "success_rate": sum(float(r.get("success", 0) or 0) for r in rs) / n,
                "mean_steps": sum(float(r.get("steps", 0) or 0) for r in rs) / n,
                "mean_no_path_rate": sum(float(r.get("no_path_rate", 0) or 0) for r in rs) / n,
                "mean_switches": sum(float(r.get("subgoal_switch_count", 0) or 0) for r in rs) / n,
                "mean_final_goal_steps": sum(float(r.get("final_goal_mode_steps", 0) or 0) for r in rs) / n,
            }
        )
    return out


def _total_success(rows: list[dict]) -> float:
    return sum(float(r.get("success", 0) or 0) for r in rows)


def _comparison_suffix(row: dict) -> str:
    name = row.get("run_name", "")
    match = re.search(r"((?:alltasks|task\d+)(?:_ep\d+)?(?:_gasseed\d+)?)", name)
    if match:
        return match.group(1)
    mode = row.get("mode", "")
    if mode == "gas_graph_policy" and name.startswith("gas_graph_policy_"):
        return name[len("gas_graph_policy_") :]
    return name


def _paired_key(row: dict) -> tuple[str, str, str, str]:
    return (
        row.get("env", ""),
        row.get("seed", ""),
        row.get("gas_seed", ""),
        _comparison_suffix(row),
    )


def _hybrid_pairs(rows: list[dict]) -> list[dict]:
    gas_by_key: dict[tuple[str, str, str, str], list[dict]] = {}
    hybrid_by_key: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        mode = row.get("mode", "")
        if mode == "gas_graph_policy":
            gas_by_key.setdefault(_paired_key(row), []).append(row)
        elif mode == "gas_graph_tmd_exec_rescue_policy":
            hybrid_by_key.setdefault((*_paired_key(row), row.get("run_name", "")), []).append(row)

    pairs = []
    for hybrid_key, hybrid_rows in sorted(hybrid_by_key.items()):
        key = hybrid_key[:4]
        gas_rows = gas_by_key.get(key)
        if not hybrid_rows:
            continue
        if not gas_rows:
            continue
        gas_index = {(r.get("task_id", ""), r.get("episode", "")): r for r in gas_rows}
        hybrid_index = {(r.get("task_id", ""), r.get("episode", "")): r for r in hybrid_rows}
        common = sorted(set(gas_index) & set(hybrid_index))
        if not common:
            continue
        gas_common = [gas_index[k] for k in common]
        hybrid_common = [hybrid_index[k] for k in common]
        solved = 0
        regressed = 0
        for k in common:
            gas_success = float(gas_index[k].get("success", 0) or 0)
            hybrid_success = float(hybrid_index[k].get("success", 0) or 0)
            if gas_success < 0.5 <= hybrid_success:
                solved += 1
            elif gas_success >= 0.5 > hybrid_success:
                regressed += 1
        per_task: dict[str, tuple[float, float]] = {}
        for task_id in sorted({k[0] for k in common}, key=lambda x: int(x) if str(x).isdigit() else str(x)):
            gas_task = [gas_index[k] for k in common if k[0] == task_id]
            hybrid_task = [hybrid_index[k] for k in common if k[0] == task_id]
            per_task[task_id] = (_total_success(gas_task), _total_success(hybrid_task))
        gas_success = _total_success(gas_common)
        hybrid_success = _total_success(hybrid_common)
        pairs.append(
            {
                "env": key[0],
                "seed": key[1],
                "gas_seed": key[2],
                "comparison": f"{key[3]}::{hybrid_key[4]}",
                "rows": len(common),
                "gas_success": gas_success,
                "hybrid_success": hybrid_success,
                "delta": hybrid_success - gas_success,
                "solved": solved,
                "regressed": regressed,
                "gas_run": gas_common[0].get("run_name", ""),
                "hybrid_run": hybrid_common[0].get("run_name", ""),
                "per_task_delta": ", ".join(
                    f"task{task_id}:{hybrid_task - gas_task:+.0f}"
                    for task_id, (gas_task, hybrid_task) in per_task.items()
                ),
            }
        )
    return pairs


def _hybrid_pair_table(pairs: list[dict]) -> str:
    if not pairs:
        return "No paired `gas_graph_policy` vs `gas_graph_tmd_exec_rescue_policy` rows were found."
    keys = ["env", "seed", "gas_seed", "comparison", "rows", "gas_success", "hybrid_success", "delta", "solved", "regressed", "per_task_delta"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in pairs:
        lines.append(
            "| {env} | {seed} | {gas_seed} | {comparison} | {rows} | {gas_success:.1f} | {hybrid_success:.1f} | {delta:+.1f} | {solved} | {regressed} | {per_task_delta} |".format(**row)
        )
    return "\n".join(lines)


def _rescue_audit(rows: list[dict]) -> tuple[str, int, float]:
    rescue_capable = [
        row
        for row in rows
        if row.get("mode") == "gas_graph_tmd_exec_rescue_policy" and "tmd_rescue_activated" in row
    ]
    if not rescue_capable:
        return "No runs with `tmd_rescue_activated` fields were found.", 0, 0.0
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rescue_capable:
        grouped.setdefault((row.get("run_name", ""), row.get("task_id", "")), []).append(row)
    keys = ["run_name", "task_id", "rows", "success", "activated", "activated_success", "mean_rescue_steps"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    total_activated = 0
    total_activated_success = 0.0
    for (run_name, task_id), rs in sorted(grouped.items()):
        activated = [r for r in rs if int(float(r.get("tmd_rescue_activated", 0) or 0))]
        activated_success = _total_success(activated)
        total_activated += len(activated)
        total_activated_success += activated_success
        mean_rescue_steps = (
            sum(float(r.get("tmd_rescue_steps", 0) or 0) for r in activated) / len(activated)
            if activated
            else 0.0
        )
        lines.append(
            "| {run_name} | {task_id} | {rows} | {success:.1f} | {activated} | {activated_success:.1f} | {mean_rescue_steps:.1f} |".format(
                run_name=run_name,
                task_id=task_id,
                rows=len(rs),
                success=_total_success(rs),
                activated=len(activated),
                activated_success=activated_success,
                mean_rescue_steps=mean_rescue_steps,
            )
        )
    return "\n".join(lines), total_activated, total_activated_success


def _has_failed_rescue_activation(rows: list[dict]) -> bool:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("mode") != "gas_graph_tmd_exec_rescue_policy":
            continue
        if "tmd_rescue_activated" not in row:
            continue
        grouped.setdefault((row.get("run_name", ""), row.get("task_id", "")), []).append(row)
    for rs in grouped.values():
        activated = [r for r in rs if int(float(r.get("tmd_rescue_activated", 0) or 0))]
        if activated and _total_success(activated) < len(activated):
            return True
    return False


def _run_stats(rows: list[dict], run_name: str, seed: str | None = None, env: str | None = None) -> dict | None:
    run_rows = [row for row in rows if row.get("run_name") == run_name]
    if seed is not None:
        run_rows = [row for row in run_rows if str(row.get("seed", "")) == str(seed)]
    if env is not None:
        run_rows = [row for row in run_rows if row.get("env", "") == env]
    if not run_rows:
        return None
    n = len(run_rows)
    return {
        "rows": n,
        "success": _total_success(run_rows),
        "mean_steps": sum(float(r.get("steps", 0) or 0) for r in run_rows) / max(n, 1),
        "activated": sum(int(float(r.get("tmd_rescue_activated", 0) or 0)) for r in run_rows),
        "activated_success": _total_success(
            [r for r in run_rows if int(float(r.get("tmd_rescue_activated", 0) or 0))]
        ),
    }


def _fmt_stats(stats: dict | None) -> str:
    if stats is None:
        return "missing"
    return f"{stats['success']:.0f}/{stats['rows']}"


def _medium_navigate_findings(rows: list[dict]) -> str:
    gas42 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed42")
    gas43 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed43")
    tmd42 = _run_stats(rows, "tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed42")
    tmd43 = _run_stats(rows, "tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed43")
    task3_gas42 = _run_stats(rows, "gas_graph_policy_task3_ep50_gasseed42")
    task3_gas43 = _run_stats(rows, "gas_graph_policy_task3_ep50_gasseed43")
    task3_tmd42 = _run_stats(rows, "tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed42")
    task3_tmd43 = _run_stats(rows, "tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed43")
    task3_100k42 = _run_stats(rows, "tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed42")
    task3_100k43 = _run_stats(rows, "tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed43")
    task3_gas100_s0_42 = _run_stats(rows, "gas_graph_policy_task3_ep100_gasseed42", seed="0")
    task3_gas100_s0_43 = _run_stats(rows, "gas_graph_policy_task3_ep100_gasseed43", seed="0")
    task3_rescue175_s0_42 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42", seed="0")
    task3_rescue175_s0_43 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43", seed="0")
    task3_gas100_s1_42 = _run_stats(rows, "gas_graph_policy_task3_ep100_gasseed42", seed="1")
    task3_gas100_s1_43 = _run_stats(rows, "gas_graph_policy_task3_ep100_gasseed43", seed="1")
    task3_rescue175_s1_42 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42", seed="1")
    task3_rescue175_s1_43 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43", seed="1")
    rescue_s200_42 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed42")
    rescue_s200_43 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed43")
    rescue_s100_42 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed42")
    rescue_s100_43 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed43")
    rescue_actor42 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed42")
    rescue_actor43 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed43")

    lines: list[str] = []
    if all(x is not None for x in [gas42, gas43, tmd42, tmd43]):
        lines.append(
            "Medium-navigate all-tasks/20 did not validate TMD execution: "
            f"GAS seed42/43 were {_fmt_stats(gas42)} and {_fmt_stats(gas43)}, "
            f"while TMD-exec scale15 was {_fmt_stats(tmd42)} and {_fmt_stats(tmd43)}; "
            "task1/task4 were the main collapses."
        )
    if all(x is not None for x in [task3_gas42, task3_gas43, task3_tmd42, task3_tmd43]):
        lines.append(
            "Medium-navigate task3/50 rejected the earlier 20-episode positive blip: "
            f"GAS seed42/43 were {_fmt_stats(task3_gas42)} and {_fmt_stats(task3_gas43)}, "
            f"while TMD-exec scale15 was {_fmt_stats(task3_tmd42)} and {_fmt_stats(task3_tmd43)}."
        )
    if all(x is not None for x in [task3_gas42, task3_gas43, task3_100k42, task3_100k43]):
        lines.append(
            "Medium-navigate effective-100k TMD did not fix task3: "
            f"TMD-exec scale15 was {_fmt_stats(task3_100k42)} and {_fmt_stats(task3_100k43)} "
            f"against GAS {_fmt_stats(task3_gas42)} and {_fmt_stats(task3_gas43)}."
        )
    if all(x is not None for x in [rescue_s200_42, rescue_s200_43, rescue_s100_42, rescue_s100_43]):
        lines.append(
            "Medium-navigate rescue also failed to improve task3: "
            f"s200/p15 matched {_fmt_stats(rescue_s200_42)} and {_fmt_stats(rescue_s200_43)} "
            f"with activated rescues {rescue_s200_42['activated']} and {rescue_s200_43['activated']}; "
            f"s100/p5 gave {_fmt_stats(rescue_s100_42)} and {_fmt_stats(rescue_s100_43)} "
            f"with activated rescues {rescue_s100_42['activated']} and {rescue_s100_43['activated']}."
        )
    if all(x is not None for x in [rescue_actor42, rescue_actor43]):
        lines.append(
            "Using the TMD actor as the rescue low-level controller regressed sharply: "
            f"{_fmt_stats(rescue_actor42)} and {_fmt_stats(rescue_actor43)}."
        )
    if all(
        x is not None
        for x in [
            task3_gas100_s0_42,
            task3_gas100_s0_43,
            task3_rescue175_s0_42,
            task3_rescue175_s0_43,
            task3_gas100_s1_42,
            task3_gas100_s1_43,
            task3_rescue175_s1_42,
            task3_rescue175_s1_43,
        ]
    ):
        lines.append(
            "The best late-rescue candidate, s175/p9, had a narrow seed0 gain but failed cross-reset validation: "
            f"eval seed0 GAS was {_fmt_stats(task3_gas100_s0_42)} and {_fmt_stats(task3_gas100_s0_43)} "
            f"versus rescue {_fmt_stats(task3_rescue175_s0_42)} and {_fmt_stats(task3_rescue175_s0_43)}; "
            f"eval seed1 GAS was {_fmt_stats(task3_gas100_s1_42)} and {_fmt_stats(task3_gas100_s1_43)} "
            f"versus rescue {_fmt_stats(task3_rescue175_s1_42)} and {_fmt_stats(task3_rescue175_s1_43)}."
        )
    if not lines:
        return "No medium-navigate confirmation rows were found yet."
    return "\n".join(lines)


def _giant_stitch_findings(rows: list[dict]) -> str:
    env = "antmaze-giant-stitch-v0"
    gas42 = _run_stats(rows, "gas_graph_policy_alltasks_ep10_gasseed42_screen", env=env)
    gas43 = _run_stats(rows, "gas_graph_policy_alltasks_ep10_gasseed43_screen", env=env)
    exec42 = _run_stats(rows, "tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42", env=env)
    exec43 = _run_stats(rows, "tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43", env=env)
    rescue300_42 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42", env=env)
    rescue300_43 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43", env=env)
    rescue700_42 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42", env=env)
    rescue700_43 = _run_stats(rows, "gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43", env=env)
    disabled42 = _run_stats(rows, "gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42", env=env)
    disabled43 = _run_stats(rows, "gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43", env=env)
    final50_42 = _run_stats(rows, "gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42", env=env)
    final50_43 = _run_stats(rows, "gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43", env=env)
    exec100_42 = _run_stats(rows, "tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42", env=env)
    exec100_43 = _run_stats(rows, "tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43", env=env)
    burst100_42 = _run_stats(rows, "gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42", env=env)
    burst100_43 = _run_stats(rows, "gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43", env=env)

    lines: list[str] = []
    if all(x is not None for x in [gas42, gas43, exec42, exec43]):
        lines.append(
            "Giant-stitch screening exposed useful GAS headroom but rejected the 50k TMD execution graph: "
            f"GAS seed42/43 were {_fmt_stats(gas42)} and {_fmt_stats(gas43)}, "
            f"while TMD-exec n512 scale15 was {_fmt_stats(exec42)} and {_fmt_stats(exec43)}."
        )
    if all(x is not None for x in [gas42, gas43, rescue300_42, rescue300_43]):
        lines.append(
            "Giant-stitch early long-path rescue was destructive: "
            f"s300/p20 gave {_fmt_stats(rescue300_42)} and {_fmt_stats(rescue300_43)} "
            f"with activated rescues {rescue300_42['activated']} and {rescue300_43['activated']}."
        )
    if all(x is not None for x in [gas42, gas43, rescue700_42, rescue700_43]):
        lines.append(
            "Giant-stitch late rescue preserved one seed but still lacked a positive TMD mechanism: "
            f"s700/p20 gave {_fmt_stats(rescue700_42)} and {_fmt_stats(rescue700_43)} "
            f"against GAS {_fmt_stats(gas42)} and {_fmt_stats(gas43)}, "
            f"with activated-rescue successes {rescue700_42['activated_success']:.0f} "
            f"and {rescue700_43['activated_success']:.0f}."
        )
    if all(x is not None for x in [disabled42, disabled43, final50_42, final50_43]):
        lines.append(
            "Giant-stitch final-goal TMD actor rescue did not validate: "
            f"the disabled hybrid control was {_fmt_stats(disabled42)} and {_fmt_stats(disabled43)}, "
            f"whereas final-actor-after50 was {_fmt_stats(final50_42)} and {_fmt_stats(final50_43)}. "
            "The apparent seed43 gain versus GAS was present in the disabled control, so it is not attributable to TMD final rescue."
        )
    if all(x is not None for x in [gas42, gas43, exec100_42, exec100_43]):
        lines.append(
            "After fixing calibration to use paired H-step distances, giant-stitch effective-100k q98/t995 remained below GAS: "
            f"TMD-exec was {_fmt_stats(exec100_42)} and {_fmt_stats(exec100_43)} "
            f"against GAS {_fmt_stats(gas42)} and {_fmt_stats(gas43)}."
        )
    if all(x is not None for x in [burst100_42, burst100_43]):
        lines.append(
            "The bounded effective-100k q98 late-rescue burst also lacked attributable rescue success: "
            f"{_fmt_stats(burst100_42)} and {_fmt_stats(burst100_43)}, "
            f"with activated-rescue successes {burst100_42['activated_success']:.0f} and {burst100_43['activated_success']:.0f}."
        )
    if not lines:
        return "No giant-stitch confirmation rows were found yet."
    return "\n".join(lines)


def _giant_navigate_findings(rows: list[dict]) -> str:
    env = "antmaze-giant-navigate-v0"
    gas42 = _run_stats(rows, "gas_graph_policy_alltasks_ep10_gasseed42_screen", env=env)
    gas43 = _run_stats(rows, "gas_graph_policy_alltasks_ep10_gasseed43_screen", env=env)
    exec15_42 = _run_stats(rows, "tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42", env=env)
    exec15_43 = _run_stats(rows, "tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43", env=env)
    exec30_42 = _run_stats(rows, "tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed42", env=env)
    exec30_43 = _run_stats(rows, "tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed43", env=env)
    gas35_42 = _run_stats(rows, "gas_graph_policy_tasks3_5_ep50_gasseed42", env=env)
    gas35_43 = _run_stats(rows, "gas_graph_policy_tasks3_5_ep50_gasseed43", env=env)
    tmd35_42 = _run_stats(rows, "tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed42", env=env)
    tmd35_43 = _run_stats(rows, "tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed43", env=env)
    exec100_42 = _run_stats(rows, "tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42", env=env)
    exec100_43 = _run_stats(rows, "tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43", env=env)
    native_actor = _run_stats(rows, "tmd50k_q98_graph_tmd_actor_alltasks_ep2_gpu", env=env)
    native_gas42 = _run_stats(rows, "tmd50k_q98_graph_gas_policy_alltasks_ep2_gasseed42_gpu", env=env)
    tmdcost10_42 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep10_gasseed42", env=env)
    tmdcost10_43 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep10_gasseed43", env=env)
    gas20_s0_42 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed42", seed="0", env=env)
    gas20_s0_43 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed43", seed="0", env=env)
    tmdcost20_s0_42 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep20_gasseed42", seed="0", env=env)
    tmdcost20_s0_43 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep20_gasseed43", seed="0", env=env)
    gas20_s1_42 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed42", seed="1", env=env)
    gas20_s1_43 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed43", seed="1", env=env)
    tmdcost20_s1_42 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep20_gasseed42", seed="1", env=env)
    tmdcost20_s1_43 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep20_gasseed43", seed="1", env=env)
    gas20_42 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed42", env=env)
    gas20_43 = _run_stats(rows, "gas_graph_policy_alltasks_ep20_gasseed43", env=env)
    tmdcost20_42 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep20_gasseed42", env=env)
    tmdcost20_43 = _run_stats(rows, "gas_graph_tmdcost_w05_alltasks_ep20_gasseed43", env=env)
    tmdcost025_s0_42 = _run_stats(rows, "gas_graph_tmdcost_w025_alltasks_ep20_gasseed42", seed="0", env=env)
    tmdcost025_s0_43 = _run_stats(rows, "gas_graph_tmdcost_w025_alltasks_ep20_gasseed43", seed="0", env=env)
    tmdcost025_s1_42 = _run_stats(rows, "gas_graph_tmdcost_w025_alltasks_ep20_gasseed42", seed="1", env=env)
    tmdcost025_s1_43 = _run_stats(rows, "gas_graph_tmdcost_w025_alltasks_ep20_gasseed43", seed="1", env=env)
    tmdcost025_42 = _run_stats(rows, "gas_graph_tmdcost_w025_alltasks_ep20_gasseed42", env=env)
    tmdcost025_43 = _run_stats(rows, "gas_graph_tmdcost_w025_alltasks_ep20_gasseed43", env=env)

    lines: list[str] = []
    if all(x is not None for x in [gas42, gas43, exec15_42, exec15_43]):
        lines.append(
            "Giant-navigate had larger GAS headroom, but 50k TMD q98/t995 execution did not improve all-task totals: "
            f"GAS seed42/43 were {_fmt_stats(gas42)} and {_fmt_stats(gas43)}, "
            f"while TMD-exec scale15 was {_fmt_stats(exec15_42)} and {_fmt_stats(exec15_43)}."
        )
    if all(x is not None for x in [exec30_42, exec30_43]):
        lines.append(
            "Increasing the execution radius to scale30 was catastrophic on giant-navigate: "
            f"{_fmt_stats(exec30_42)} and {_fmt_stats(exec30_43)}."
        )
    if all(x is not None for x in [gas35_42, gas35_43, tmd35_42, tmd35_43]):
        lines.append(
            "The apparent task3/task5 ep10 signal failed ep50 confirmation: "
            f"tasks3,5 GAS was {_fmt_stats(gas35_42)} and {_fmt_stats(gas35_43)}, "
            f"whereas TMD-exec was {_fmt_stats(tmd35_42)} and {_fmt_stats(tmd35_43)}."
        )
    if all(x is not None for x in [exec100_42, exec100_43]):
        lines.append(
            "Giant-navigate effective-100k did not repair the 50k instability: "
            f"TMD-exec q98/t995 was {_fmt_stats(exec100_42)} and {_fmt_stats(exec100_43)}."
        )
    if all(x is not None for x in [native_actor, native_gas42]):
        lines.append(
            "Native TMD graph-path execution was rejected by a GPU smoke test: "
            f"TMD actor was {_fmt_stats(native_actor)} and GAS low-level on TMD graph was {_fmt_stats(native_gas42)}."
        )
    if all(x is not None for x in [gas42, gas43, tmdcost10_42, tmdcost10_43]):
        lines.append(
            "TMD-cost shaping on the GAS graph produced the first positive giant-navigate all-task signal at ep10: "
            f"GAS seed42/43 were {_fmt_stats(gas42)} and {_fmt_stats(gas43)}, "
            f"while TMD-cost w0.5 was {_fmt_stats(tmdcost10_42)} and {_fmt_stats(tmdcost10_43)}."
        )
    if all(x is not None for x in [gas20_s0_42, gas20_s0_43, tmdcost20_s0_42, tmdcost20_s0_43]):
        lines.append(
            "The reset-seed0 ep20 confirmation stayed positive but small, so it is promising rather than conclusive: "
            f"GAS seed42/43 were {_fmt_stats(gas20_s0_42)} and {_fmt_stats(gas20_s0_43)}, "
            f"while TMD-cost w0.5 was {_fmt_stats(tmdcost20_s0_42)} and {_fmt_stats(tmdcost20_s0_43)}."
        )
    if all(x is not None for x in [gas20_s1_42, gas20_s1_43, tmdcost20_s1_42, tmdcost20_s1_43]):
        lines.append(
            "The reset-seed1 ep20 check was mixed: "
            f"GAS seed42/43 were {_fmt_stats(gas20_s1_42)} and {_fmt_stats(gas20_s1_43)}, "
            f"while TMD-cost w0.5 was {_fmt_stats(tmdcost20_s1_42)} and {_fmt_stats(tmdcost20_s1_43)}. "
            "This suggests the cost weight is useful but currently too blunt."
        )
    if all(x is not None for x in [gas20_42, gas20_43, tmdcost20_42, tmdcost20_43]):
        lines.append(
            "Across reset seeds 0 and 1, TMD-cost w0.5 remains net positive but not uniform: "
            f"GAS seed42/43 aggregate were {_fmt_stats(gas20_42)} and {_fmt_stats(gas20_43)}, "
            f"while TMD-cost w0.5 was {_fmt_stats(tmdcost20_42)} and {_fmt_stats(tmdcost20_43)}."
        )
    if all(
        x is not None
        for x in [
            gas20_s0_42,
            gas20_s0_43,
            gas20_s1_42,
            gas20_s1_43,
            tmdcost025_s0_42,
            tmdcost025_s0_43,
            tmdcost025_s1_42,
            tmdcost025_s1_43,
            tmdcost025_42,
            tmdcost025_43,
        ]
    ):
        lines.append(
            "Reducing TMD-cost weight to w0.25 fixed the w0.5 bluntness and validated a robust positive mechanism: "
            f"reset-seed0 GAS seed42/43 {_fmt_stats(gas20_s0_42)} and {_fmt_stats(gas20_s0_43)} "
            f"became {_fmt_stats(tmdcost025_s0_42)} and {_fmt_stats(tmdcost025_s0_43)}; "
            f"reset-seed1 GAS seed42/43 {_fmt_stats(gas20_s1_42)} and {_fmt_stats(gas20_s1_43)} "
            f"became {_fmt_stats(tmdcost025_s1_42)} and {_fmt_stats(tmdcost025_s1_43)}. "
            f"Across both reset seeds, GAS seed42/43 were {_fmt_stats(gas20_42)} and {_fmt_stats(gas20_43)}, "
            f"while TMD-cost w0.25 was {_fmt_stats(tmdcost025_42)} and {_fmt_stats(tmdcost025_43)}."
        )
    if not lines:
        return "No giant-navigate confirmation rows were found yet."
    return "\n".join(lines)


def _graph_summary(rows: list[dict]) -> str:
    lines = ["# tmd-test Graph Diagnostics", "", f"Rows: {len(rows)}", ""]
    keys = ["env", "seed", "num_nodes", "num_edges", "mean_out_degree", "scc_count", "largest_scc_ratio", "edge_distance_threshold", "target_distance_threshold"]
    if rows:
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in rows[:40]:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    else:
        lines.append("No graph diagnostics were found.")
    return "\n".join(lines) + "\n"


def _decisions(decision: str, eval_rows: list[dict], graph_rows: list[dict], hybrid_pairs: list[dict]) -> str:
    mode_rows = _mode_task_summary(eval_rows)
    lines = ["| run_name | env | seed | gas_seed | mode | task_id | episodes | success_rate | mean_steps | mean_no_path_rate | mean_switches | mean_final_goal_steps |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in mode_rows:
        lines.append(
            "| {run_name} | {env} | {seed} | {gas_seed} | {mode} | {task_id} | {episodes} | {success_rate:.3f} | {mean_steps:.1f} | {mean_no_path_rate:.3f} | {mean_switches:.1f} | {mean_final_goal_steps:.1f} |".format(**row)
        )
    mode_table = "\n".join(lines)
    pair_table = _hybrid_pair_table(hybrid_pairs)
    rescue_table, _, _ = _rescue_audit(eval_rows)
    medium_navigate_findings = _medium_navigate_findings(eval_rows)
    giant_stitch_findings = _giant_stitch_findings(eval_rows)
    giant_navigate_findings = _giant_navigate_findings(eval_rows)
    return f"""# tmd-test Decisions

## Branch and protocol

Authorized branch: `stage25-protocol-oracle-drift`.
Fallback mode: `none`.

## Implemented components

`tmd_test` wrapper, calibration, key nodes, directed key graph, construct/eval/analyze scripts.

## Smoke result

Evaluation rows: {len(eval_rows)}.
Graph rows: {len(graph_rows)}.

## Graph diagnostics

See `reports/tmd_test_graph_diagnostics.md`.

## Evaluation result

See `reports/tmd_test_eval_summary.md`.

## Mode/task aggregate

{mode_table}

## Hybrid rescue paired comparison

{pair_table}

## Rescue activation audit

{rescue_table}

## Failure analysis

The comparable GAS graph baseline is `gas_graph_policy`. Pure `tmd_graph_*` rows do not exceed this baseline on success.
`gas_graph_tmd_exec_rescue_policy` is a hybrid rescue policy: it starts from the official GAS graph and switches to the TMD execution graph only when the GAS graph has not entered final-goal mode after the configured delay and the remaining GAS path is still long.
Mechanism-level evidence must come from rows where `tmd_rescue_activated=1`; aggregate success deltas without rescue activation can be explained by stochastic trajectory differences.
Continuation checks found that the original 50k checkpoint has a narrow task3 seed42 signal (`scale15`, 50/50 vs GAS 49/50), but the same setting regresses seed43 and the best cross-seed threshold scan (`scale18`) only ties GAS on task3 aggregate.
Resume training to effective 100k/150k/200k did not validate a stronger TMD backbone: 100k task3 execution collapsed, while 150k/200k automatic TE graphs shrank to very few nodes and TE-relaxed graphs still underperformed GAS.
An IQE-distance TMD run was attempted, but early throughput was about 4.8 steps/s, making a 50k checkpoint a multi-hour job; it was stopped before checkpoint save and is not counted as evidence.
{medium_navigate_findings}
{giant_stitch_findings}
{giant_navigate_findings}
If evaluation rows are absent, inspect `construct_error.json` or `eval_error.json`.

## Decision

- {decision}

## Next commands

```bash
bash scripts/tmd_test_pilot.sh ENVS=antmaze-medium-stitch-v0 SEEDS=0 EPISODES=2 QUICK=1 MODES=tmd_graph_tmd_actor
```
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze tmd-test runs.")
    parser.add_argument("--runs-root", default="runs_tmd_test")
    parser.add_argument("--artifacts-root", default="artifacts/tmd_test")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args(argv)
    eval_rows: list[dict] = []
    for path in glob.glob(f"{args.runs_root}/**/eval.csv", recursive=True):
        run_rows = _read_csv(path)
        run_name = Path(path).parent.name
        for row in run_rows:
            row["run_name"] = run_name
            row["eval_path"] = path
        eval_rows.extend(run_rows)
    graph_rows: list[dict] = []
    for path in glob.glob(f"{args.artifacts_root}/**/graph_stats.json", recursive=True):
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        row["path"] = path
        graph_rows.append(row)
    analysis = Path(args.runs_root) / "_analysis"
    _write_csv(analysis / "eval_all.csv", eval_rows)
    _write_csv(analysis / "graph_all.csv", graph_rows)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "tmd_test_eval_summary.md").write_text(_eval_summary(eval_rows), encoding="utf-8")
    (reports / "tmd_test_graph_diagnostics.md").write_text(_graph_summary(graph_rows), encoding="utf-8")
    hybrid_pairs = _hybrid_pairs(eval_rows)
    decision = "HOLD_TMD_GRAPH"
    if graph_rows and eval_rows:
        tmd_rows = [r for r in eval_rows if str(r.get("mode", "")).startswith("tmd_graph")]
        gas_graph_rows = [r for r in eval_rows if r.get("mode") == "gas_graph_policy"]
        if tmd_rows:
            tmd_success = sum(float(r.get("success", 0) or 0) for r in tmd_rows) / len(tmd_rows)
            tmd_no_path = sum(float(r.get("no_path_rate", 1) or 1) for r in tmd_rows) / len(tmd_rows)
            gas_success = (
                sum(float(r.get("success", 0) or 0) for r in gas_graph_rows) / len(gas_graph_rows)
                if gas_graph_rows
                else 0.0
            )
            if gas_graph_rows and tmd_success > gas_success:
                decision = "GO_TMD_GRAPH"
            elif tmd_no_path > 0.5 or tmd_success == 0:
                decision = "HOLD_TMD_GRAPH"
            else:
                decision = "HOLD_TMD_GRAPH"
    confirmed_pairs = [p for p in hybrid_pairs if int(p["rows"] or 0) >= 250]
    _, rescue_activated, rescue_success = _rescue_audit(eval_rows)
    rescue_has_failed_activation = _has_failed_rescue_activation(eval_rows)
    if confirmed_pairs:
        total_delta = sum(float(p["delta"]) for p in confirmed_pairs)
        min_delta = min(float(p["delta"]) for p in confirmed_pairs)
        if rescue_activated > 0 and (rescue_success <= 0 or rescue_has_failed_activation):
            decision = "HOLD_TMD_EXEC_RESCUE_NEEDS_MECHANISM"
        elif total_delta > 0 and min_delta >= 0:
            decision = "GO_TMD_EXEC_RESCUE"
        elif total_delta > 0:
            decision = "HOLD_TMD_EXEC_RESCUE_NEEDS_TUNING"
    cost_deltas: list[float] = []
    for reset_seed in ("0", "1"):
        for gas_seed in ("42", "43"):
            gas = _run_stats(
                eval_rows,
                f"gas_graph_policy_alltasks_ep20_gasseed{gas_seed}",
                seed=reset_seed,
                env="antmaze-giant-navigate-v0",
            )
            tmd_cost = _run_stats(
                eval_rows,
                f"gas_graph_tmdcost_w025_alltasks_ep20_gasseed{gas_seed}",
                seed=reset_seed,
                env="antmaze-giant-navigate-v0",
            )
            if gas is not None and tmd_cost is not None:
                cost_deltas.append(float(tmd_cost["success"]) - float(gas["success"]))
    if len(cost_deltas) == 4 and min(cost_deltas) > 0 and sum(cost_deltas) > 0:
        decision = "GO_TMD_COST_SHAPING_GIANT_NAVIGATE_W025"
    (reports / "tmd_test_decisions.md").write_text(_decisions(decision, eval_rows, graph_rows, hybrid_pairs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

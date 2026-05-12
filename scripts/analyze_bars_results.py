#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

DECISIONS = [
    'GO',
    'RETRY_STAGE',
    'FIX_REACHABILITY',
    'FIX_BOUNDARY',
    'FIX_GRAPH',
    'FIX_LOW_LEVEL_POLICY',
    'EXPAND_EVAL',
]

REPORT_DEFAULTS = {
    'stage1': 'reports/stage1_diagnostics.md',
    'stage2': 'reports/stage2_node_ablation.md',
    'stage3': 'reports/stage3_quick_eval.md',
}

FAILURE_PATTERNS = [
    ('dataset_truncated', re.compile(r'truncated file|Failed to prepare D4RL dataset', re.I)),
    ('oom', re.compile(r'CUDA out of memory|OutOfMemoryError', re.I)),
    ('no_candidate_edges', re.compile(r'No candidate edges constructed', re.I)),
    ('traceback', re.compile(r'Traceback', re.I)),
]


def _maybe_collect(log_root: Path) -> None:
    analysis_dir = log_root / '_analysis'
    summary_path = analysis_dir / 'summary_all.csv'
    if summary_path.exists():
        return
    subprocess.run([sys.executable, 'scripts/collect_csv.py', '--log-root', str(log_root)], check=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def _manifest_records(log_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for manifest in log_root.glob('**/manifest.json'):
        try:
            rows.append(pd.read_json(manifest, typ='series').to_dict())
        except Exception:
            continue
    return pd.DataFrame(rows)


def _summary_run_records(summary_all: pd.DataFrame) -> pd.DataFrame:
    if summary_all.empty or 'run_id' not in summary_all.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for run_id, g in summary_all.groupby('run_id', dropna=False):
        g = g.sort_values('time_sec') if 'time_sec' in g.columns else g
        last = g.iloc[-1].to_dict()
        rows.append(last)
    return pd.DataFrame(rows)


def _latest_metric_rows(df: pd.DataFrame, phase: str, group_cols: List[str]) -> pd.DataFrame:
    if df.empty or 'phase' not in df.columns:
        return pd.DataFrame()
    sub = df[df['phase'] == phase].copy()
    if sub.empty:
        return sub
    sort_cols = ['time_sec'] if 'time_sec' in sub.columns else []
    if sort_cols:
        sub = sub.sort_values(sort_cols)
    if not group_cols:
        return sub.tail(1)
    return sub.groupby(group_cols, dropna=False, as_index=False).tail(1)


def _planner_summary(path_diag: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if path_diag.empty:
        return pd.DataFrame()
    agg = (
        path_diag.groupby(keys, dropna=False)
        .agg(
            pairs=('pair_id', 'count'),
            found_rate=('found', 'mean'),
            total_risk=('total_risk', 'mean'),
            total_boundary=('total_boundary', 'mean'),
            total_cost=('total_cost', 'mean'),
            objective=('objective', 'mean'),
            num_edges=('num_edges', 'mean'),
            num_subgoals=('num_subgoals', 'mean'),
        )
        .reset_index()
    )
    return agg.sort_values(keys).reset_index(drop=True)


def _edge_summary(edge_diag: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if edge_diag.empty:
        return pd.DataFrame()
    keep = [
        'reach_auc_proxy',
        'reach_auprc_proxy',
        'false_positive_proxy_rate',
        'reachable_edge_coverage_proxy',
        'selected_edges',
        'num_edges',
    ]
    agg = edge_diag.groupby(keys, dropna=False)[[c for c in keep if c in edge_diag.columns]].mean(numeric_only=True).reset_index()
    return agg.sort_values(keys).reset_index(drop=True)


def _boundary_summary(boundary_diag: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if boundary_diag.empty:
        return pd.DataFrame()
    keep = ['psi_mean', 'psi_p10', 'psi_p50', 'psi_p90', 'supported_pair_rate', 'num_pairs']
    agg = boundary_diag.groupby(keys, dropna=False)[[c for c in keep if c in boundary_diag.columns]].mean(numeric_only=True).reset_index()
    return agg.sort_values(keys).reset_index(drop=True)


def _eval_summary(eval_df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    keep = ['success', 'return', 'steps', 'replans', 'no_path_count', 'last_plan_edges', 'goal_distance_final']
    agg = (
        eval_df.groupby(keys, dropna=False)[[c for c in keep if c in eval_df.columns]]
        .agg(['mean', 'std'])
        .reset_index()
    )
    agg.columns = ['_'.join([str(p) for p in col if str(p) != '']).rstrip('_') for col in agg.columns.to_flat_index()]
    return agg.sort_values(keys).reset_index(drop=True)


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
    return out


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '_No data._'
    safe = df.fillna('NaN').astype(object)
    cols = list(safe.columns)
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    rows = ['| ' + ' | '.join(str(safe.iloc[i][c]) for c in cols) + ' |' for i in range(len(safe))]
    return '\n'.join([header, sep, *rows])


def _stderr_failure_modes(log_root: Path) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for stderr_path in log_root.glob('**/stderr.log'):
        try:
            text = stderr_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        matched = False
        for label, pattern in FAILURE_PATTERNS:
            if pattern.search(text):
                counter[label] += 1
                matched = True
                break
        if not matched and 'Traceback' in text:
            counter['traceback'] += 1
    rows = [{'failure_mode': k, 'count': v} for k, v in counter.most_common()]
    return pd.DataFrame(rows)


def _decision_stage1(run_counts: Dict[str, int], edge_summary: pd.DataFrame, planner_summary: pd.DataFrame) -> str:
    if not edge_summary.empty and 'reach_auc_proxy' in edge_summary.columns:
        best_auc = pd.to_numeric(edge_summary['reach_auc_proxy'], errors='coerce').max()
        if pd.notna(best_auc) and best_auc < 0.60:
            return 'FIX_REACHABILITY'
    if run_counts.get('failed', 0) > 0:
        return 'RETRY_STAGE'
    if planner_summary.empty:
        return 'RETRY_STAGE'
    planner = planner_summary.copy()
    if 'variant' not in planner.columns:
        return 'RETRY_STAGE'
    shortest = planner[planner['variant'] == 'shortest']
    full_bars = planner[planner['variant'] == 'full_bars']
    reach = planner[planner['variant'] == 'reachability']
    if full_bars.empty or shortest.empty:
        return 'RETRY_STAGE'
    merged = full_bars.merge(shortest, on=['env'], suffixes=('_bars', '_shortest'))
    if 'found_rate_bars' in merged.columns and 'found_rate_shortest' in merged.columns:
        if ((merged['found_rate_bars'] + 0.15) < merged['found_rate_shortest']).any():
            return 'FIX_GRAPH'
    if 'total_boundary_bars' in merged.columns and 'total_risk_bars' in merged.columns:
        if 'total_risk_shortest' in merged.columns and (merged['total_risk_bars'] <= merged['total_risk_shortest']).all():
            return 'GO'
    if not reach.empty:
        mr = reach.merge(full_bars, on=['env'], suffixes=('_reach', '_bars'))
        if 'total_risk_reach' in mr.columns and 'total_risk_bars' in mr.columns and (mr['total_risk_bars'] > mr['total_risk_reach']).all():
            return 'FIX_BOUNDARY'
    return 'RETRY_STAGE'


def _decision_stage2(run_counts: Dict[str, int], planner_summary: pd.DataFrame, edge_summary: pd.DataFrame) -> str:
    if run_counts.get('failed', 0) > 0:
        return 'RETRY_STAGE'
    if planner_summary.empty or 'node_method' not in planner_summary.columns:
        return 'RETRY_STAGE'
    full = planner_summary.copy()
    if 'variant' in full.columns:
        full = full[full['variant'] == 'full_bars'].copy()
    if full.empty:
        return 'RETRY_STAGE'
    bars = full[full['node_method'] == 'bars']
    if bars.empty:
        return 'RETRY_STAGE'
    score_cols = [c for c in ['found_rate', 'total_risk', 'total_boundary'] if c in full.columns]
    if not score_cols:
        return 'RETRY_STAGE'
    full['rank_found'] = full.groupby('env')['found_rate'].rank(method='min', ascending=False) if 'found_rate' in full.columns else math.nan
    full['rank_risk'] = full.groupby('env')['total_risk'].rank(method='min', ascending=True) if 'total_risk' in full.columns else math.nan
    full['rank_boundary'] = full.groupby('env')['total_boundary'].rank(method='min', ascending=True) if 'total_boundary' in full.columns else math.nan
    bars_rank = full[full['node_method'] == 'bars'][['rank_found', 'rank_risk', 'rank_boundary']].mean(numeric_only=True).mean()
    best_rank = full.groupby('node_method')[['rank_found', 'rank_risk', 'rank_boundary']].mean(numeric_only=True).mean(axis=1).min()
    if pd.notna(bars_rank) and pd.notna(best_rank) and bars_rank <= best_rank + 0.25:
        return 'GO'
    if not edge_summary.empty and 'reach_auc_proxy' in edge_summary.columns and pd.to_numeric(edge_summary['reach_auc_proxy'], errors='coerce').mean() < 0.60:
        return 'FIX_REACHABILITY'
    return 'FIX_GRAPH'


def _decision_stage3(run_counts: Dict[str, int], eval_summary: pd.DataFrame) -> str:
    if run_counts.get('failed', 0) > 0:
        return 'RETRY_STAGE'
    if eval_summary.empty:
        return 'RETRY_STAGE'
    shortest = eval_summary[eval_summary['variant'] == 'shortest']
    full_bars = eval_summary[eval_summary['variant'] == 'full_bars']
    reach = eval_summary[eval_summary['variant'] == 'reachability']
    if shortest.empty or full_bars.empty:
        return 'RETRY_STAGE'
    comp = full_bars.merge(shortest, on='env', suffixes=('_bars', '_shortest'))
    if 'success_mean_bars' in comp.columns and 'success_mean_shortest' in comp.columns:
        improved = (comp['success_mean_bars'] > comp['success_mean_shortest']).sum()
        if improved >= 2:
            return 'EXPAND_EVAL'
    comp_r = reach.merge(shortest, on='env', suffixes=('_reach', '_shortest'))
    if 'success_mean_reach' in comp_r.columns and 'success_mean_shortest' in comp_r.columns:
        improved = (comp_r['success_mean_reach'] > comp_r['success_mean_shortest']).sum()
        if improved >= 2:
            return 'EXPAND_EVAL'
    if not reach.empty and not full_bars.empty:
        rf = full_bars.merge(reach, on='env', suffixes=('_bars', '_reach'))
        if 'success_mean_bars' in rf.columns and 'success_mean_reach' in rf.columns and (rf['success_mean_bars'] < rf['success_mean_reach']).all():
            return 'FIX_BOUNDARY'
    return 'FIX_LOW_LEVEL_POLICY'


def _decision(stage: str, run_counts: Dict[str, int], edge_summary: pd.DataFrame, planner_summary: pd.DataFrame, eval_summary: pd.DataFrame) -> str:
    if stage == 'stage1':
        return _decision_stage1(run_counts, edge_summary, planner_summary)
    if stage == 'stage2':
        return _decision_stage2(run_counts, planner_summary, edge_summary)
    return _decision_stage3(run_counts, eval_summary)


def _next_actions(stage: str, decision: str, log_root: Path) -> List[str]:
    base = [f'python scripts/collect_csv.py --log-root {log_root}', f'python scripts/analyze_bars_results.py --log-root {log_root} --stage {stage}']
    extra = {
        'GO': ['Proceed to the next planned stage.'],
        'RETRY_STAGE': ['Rerun only failed or incomplete runs after the current code fixes are validated.'],
        'FIX_REACHABILITY': ['Adjust reachability negatives, features, or loss balance before expanding experiments.'],
        'FIX_BOUNDARY': ['Lower boundary penalty or switch to softer boundary costs, then rerun a small subset.'],
        'FIX_GRAPH': ['Relax graph pruning/top-k or revisit node selection and edge construction before full reruns.'],
        'FIX_LOW_LEVEL_POLICY': ['Inspect GCBC conditioning, normalization, and subgoal horizon with a small online eval subset.'],
        'EXPAND_EVAL': ['Increase eval episodes or expand to large tasks under the same protocol.'],
    }
    return base + extra.get(decision, [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-root', required=True)
    ap.add_argument('--stage', choices=['stage1', 'stage2', 'stage3'], required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    log_root = Path(args.log_root)
    out_path = Path(args.out or REPORT_DEFAULTS[args.stage])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_dir = log_root / '_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _maybe_collect(log_root)

    summary_all = _read_csv(analysis_dir / 'summary_all.csv')
    diagnostics_all = _coerce_numeric(_read_csv(analysis_dir / 'diagnostics_all.csv'), ['time_sec', 'reach_auc_proxy', 'reach_auprc_proxy', 'false_positive_proxy_rate', 'reachable_edge_coverage_proxy', 'psi_mean', 'psi_p10', 'psi_p50', 'psi_p90', 'supported_pair_rate', 'found', 'total_risk', 'total_boundary', 'total_cost', 'objective', 'num_edges', 'num_subgoals'])
    graph_all = _coerce_numeric(_read_csv(analysis_dir / 'graph_all.csv'), ['time_sec', 'num_nodes', 'num_edges', 'mean_out_degree', 'p_exec_mean', 'p_exec_p10', 'risk_mean', 'cost_mean'])
    eval_all = _coerce_numeric(_read_csv(analysis_dir / 'eval_all.csv'), ['time_sec', 'success', 'return', 'steps', 'replans', 'no_path_count', 'last_plan_edges', 'goal_distance_final'])

    manifest_df = _manifest_records(log_root)
    run_df = manifest_df if not manifest_df.empty else _summary_run_records(summary_all)
    if run_df.empty:
        run_df = pd.DataFrame(columns=['run_id', 'status'])

    status_counts = run_df['status'].fillna('unknown').value_counts().to_dict() if 'status' in run_df.columns else {}
    run_counts = {
        'total': int(len(run_df)),
        'completed': int(status_counts.get('completed', 0)),
        'failed': int(status_counts.get('failed', 0)),
        'terminated': int(status_counts.get('terminated', 0) + sum(v for k, v in status_counts.items() if str(k).startswith('stopped'))),
        'archived': int(sum(1 for _ in log_root.glob('**/archives/*.tar.gz'))),
    }

    edge_diag = _latest_metric_rows(diagnostics_all, 'edge_diag', ['run_id'])
    boundary_diag = _latest_metric_rows(diagnostics_all, 'boundary_diag', ['run_id'])
    path_diag = diagnostics_all[diagnostics_all.get('phase', pd.Series(dtype=object)) == 'path_diag'].copy()
    path_diag = path_diag[path_diag['variant'].notna()] if 'variant' in path_diag.columns else path_diag

    edge_keys = ['env']
    planner_keys = ['env', 'variant']
    boundary_keys = ['env']
    if args.stage == 'stage2':
        edge_keys = ['env', 'node_method']
        planner_keys = ['env', 'node_method', 'variant']
        boundary_keys = ['env', 'node_method']

    edge_summary = _round_df(_edge_summary(edge_diag, edge_keys))
    planner_summary = _round_df(_planner_summary(path_diag, planner_keys)) if not path_diag.empty else pd.DataFrame()
    boundary_summary = _round_df(_boundary_summary(boundary_diag, boundary_keys))
    eval_keys = ['env', 'variant']
    eval_summary = _round_df(_eval_summary(eval_all, eval_keys))

    stage_prefix = args.stage
    if not edge_summary.empty:
        edge_summary.to_csv(analysis_dir / f'{stage_prefix}_edge_summary.csv', index=False)
    if not planner_summary.empty:
        planner_summary.to_csv(analysis_dir / f'{stage_prefix}_planner_summary.csv', index=False)
        if args.stage == 'stage1':
            planner_summary.to_csv(analysis_dir / f'{stage_prefix}_summary.csv', index=False)
    if not boundary_summary.empty:
        boundary_summary.to_csv(analysis_dir / f'{stage_prefix}_boundary_summary.csv', index=False)
    if args.stage == 'stage3' and not eval_summary.empty:
        eval_summary.to_csv(analysis_dir / f'{stage_prefix}_summary.csv', index=False)

    failure_modes = _stderr_failure_modes(log_root)
    decision = _decision(args.stage, run_counts, edge_summary, planner_summary, eval_summary)
    next_actions = _next_actions(args.stage, decision, log_root)

    core_table = eval_summary if args.stage == 'stage3' else planner_summary
    diagnostics_table = edge_summary if args.stage in {'stage1', 'stage2'} else boundary_summary

    lines = [
        f'# {args.stage.capitalize()} Report',
        '',
        '## Run Completion',
        f'- total runs: {run_counts["total"]}',
        f'- completed: {run_counts["completed"]}',
        f'- failed: {run_counts["failed"]}',
        f'- terminated: {run_counts["terminated"]}',
        f'- archived: {run_counts["archived"]}',
        '',
        '## Core Metrics',
        _markdown_table(core_table),
        '',
        '## Diagnostics',
        _markdown_table(diagnostics_table),
        '',
        '## Failure Modes',
        _markdown_table(failure_modes),
        '',
        '## Decision',
        f'- decision: {decision}',
        '',
        '## Next Actions',
    ]
    lines.extend(f'- {item}' for item in next_actions)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()

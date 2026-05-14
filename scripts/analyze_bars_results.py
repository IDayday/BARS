#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

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


def _maybe_collect(log_root: Path, force: bool = False) -> None:
    analysis_dir = log_root / '_analysis'
    summary_path = analysis_dir / 'summary_all.csv'
    if summary_path.exists() and not force:
        return
    subprocess.run([sys.executable, 'scripts/collect_csv.py', '--log-root', str(log_root)], check=False)


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
            rec = pd.read_json(manifest, typ='series').to_dict()
            rec.setdefault('manifest_path', str(manifest))
            rows.append(rec)
        except Exception:
            continue
    return pd.DataFrame(rows)


def _summary_run_records(summary_all: pd.DataFrame) -> pd.DataFrame:
    if summary_all.empty or 'run_id' not in summary_all.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, g in summary_all.groupby('run_id', dropna=False):
        if 'time_sec' in g.columns:
            g = g.sort_values('time_sec')
        rows.append(g.iloc[-1].to_dict())
    return pd.DataFrame(rows)


def _latest_metric_rows(df: pd.DataFrame, phase: str, group_cols: List[str]) -> pd.DataFrame:
    if df.empty or 'phase' not in df.columns:
        return pd.DataFrame()
    sub = df[df['phase'] == phase].copy()
    if sub.empty:
        return sub
    if 'time_sec' in sub.columns:
        sub = sub.sort_values('time_sec')
    group_cols = [c for c in group_cols if c in sub.columns]
    if not group_cols:
        return sub.tail(1)
    return sub.groupby(group_cols, dropna=False, as_index=False).tail(1)


def _agg_mean(df: pd.DataFrame, keys: List[str], cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = [c for c in keys if c in df.columns]
    cols = [c for c in cols if c in df.columns]
    if not keys or not cols:
        return pd.DataFrame()
    return df.groupby(keys, dropna=False)[cols].mean(numeric_only=True).reset_index().sort_values(keys).reset_index(drop=True)


def _planner_summary(path_diag: pd.DataFrame, keys: List[str], nonzero_only: bool = False) -> pd.DataFrame:
    if path_diag.empty:
        return pd.DataFrame()
    df = path_diag.copy()
    if nonzero_only and 'num_edges' in df.columns:
        df = df[pd.to_numeric(df['num_edges'], errors='coerce').fillna(0) > 0].copy()
    keys = [c for c in keys if c in df.columns]
    if not keys:
        return pd.DataFrame()
    # Avoid counting lambda-sweep rows as extra pairs by grouping only row-level
    # means. The pair count is still useful as the number of logged rows.
    agg_spec = {}
    if 'pair_id' in df.columns:
        agg_spec['rows'] = ('pair_id', 'count')
        agg_spec['unique_pairs'] = ('pair_id', 'nunique')
    for col in ['found', 'total_risk', 'total_boundary', 'total_cost', 'objective', 'num_edges', 'num_subgoals', 'is_trivial_pair', 'lambda_risk']:
        if col in keys:
            continue
        if col in df.columns:
            out_col = 'found_rate' if col == 'found' else ('trivial_pair_rate' if col == 'is_trivial_pair' else col)
            agg_spec[out_col] = (col, 'mean')
    if not agg_spec:
        return pd.DataFrame()
    return df.groupby(keys, dropna=False).agg(**agg_spec).reset_index().sort_values(keys).reset_index(drop=True)


def _eval_summary(eval_df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    keys = [c for c in keys if c in eval_df.columns]
    keep = [c for c in ['success', 'return', 'steps', 'replans', 'no_path_count', 'initial_plan_failed_count', 'plan_failed_initial', 'fallback_used', 'fallback_count', 'direct_goal_attempts', 'last_plan_edges', 'first_plan_edges', 'max_plan_edges', 'mean_plan_edges', 'num_plan_calls', 'num_subgoal_attempts', 'num_subgoal_reached', 'subgoal_reach_rate', 'goal_distance_final', 'subgoal_horizon', 'subgoal_threshold', 'success_threshold', 'lambda_risk', 'lambda_boundary'] if c in eval_df.columns]
    if not keys or not keep:
        return pd.DataFrame()
    agg = eval_df.groupby(keys, dropna=False)[keep].agg(['mean', 'std']).reset_index()
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
    return pd.DataFrame([{'failure_mode': k, 'count': v} for k, v in counter.most_common()])


def _decision_stage1(run_counts: Dict[str, int], edge_summary: pd.DataFrame, planner_nonzero: pd.DataFrame, boundary_summary: pd.DataFrame, balanced_edge_summary: pd.DataFrame) -> str:
    if run_counts.get('failed', 0) > 0:
        return 'RETRY_STAGE'
    # Prefer balanced support metrics when present.
    auc_col = 'edge_auc_balanced' if 'edge_auc_balanced' in balanced_edge_summary.columns else 'reach_auc_proxy'
    auc_df = balanced_edge_summary if auc_col == 'edge_auc_balanced' else edge_summary
    if not auc_df.empty and auc_col in auc_df.columns:
        auc_mean = pd.to_numeric(auc_df[auc_col], errors='coerce').mean()
        if pd.notna(auc_mean) and auc_mean < 0.60:
            return 'FIX_REACHABILITY'
    if planner_nonzero.empty:
        return 'RETRY_STAGE'
    if 'variant' not in planner_nonzero.columns:
        return 'RETRY_STAGE'
    shortest = planner_nonzero[planner_nonzero['variant'] == 'shortest']
    reach = planner_nonzero[planner_nonzero['variant'] == 'reachability']
    full = planner_nonzero[planner_nonzero['variant'].isin(['full_bars', 'bars'])]
    if not reach.empty and not shortest.empty:
        m = reach.merge(shortest, on='env', suffixes=('_reach', '_shortest'))
        if 'total_risk_reach' in m.columns and 'total_risk_shortest' in m.columns:
            if (m['total_risk_reach'] < m['total_risk_shortest']).mean() >= 0.75:
                # If boundary still has no support, declare boundary not ready but
                # reachability can move to Stage 2a.
                if not boundary_summary.empty and 'supported_pair_rate' in boundary_summary.columns:
                    spr = pd.to_numeric(boundary_summary['supported_pair_rate'], errors='coerce').mean()
                    if pd.notna(spr) and spr <= 1e-6:
                        return 'FIX_BOUNDARY'
                return 'GO'
    if not full.empty and not reach.empty:
        mr = full.merge(reach, on='env', suffixes=('_bars', '_reach'))
        if 'total_risk_bars' in mr.columns and 'total_risk_reach' in mr.columns:
            if (mr['total_risk_bars'] > mr['total_risk_reach']).all():
                return 'FIX_BOUNDARY'
    return 'RETRY_STAGE'


def _decision_stage2(run_counts: Dict[str, int], planner_nonzero: pd.DataFrame, edge_summary: pd.DataFrame) -> str:
    if run_counts.get('failed', 0) > 0:
        return 'RETRY_STAGE'
    if planner_nonzero.empty or 'node_method' not in planner_nonzero.columns:
        return 'RETRY_STAGE'
    return 'GO'


def _decision_stage3(run_counts: Dict[str, int], eval_summary: pd.DataFrame) -> str:
    if run_counts.get('failed', 0) > 0:
        return 'RETRY_STAGE'
    if eval_summary.empty:
        return 'RETRY_STAGE'
    return 'EXPAND_EVAL'


def _next_actions(stage: str, decision: str, log_root: Path) -> List[str]:
    base = [f'python scripts/collect_csv.py --log-root {log_root}', f'python scripts/analyze_bars_results.py --log-root {log_root} --stage {stage}']
    extra = {
        'GO': ['Proceed to the next planned stage, but keep boundary-disabled and reachability-only variants if boundary support is weak.'],
        'RETRY_STAGE': ['Rerun only failed or incomplete runs after validating current fixes.'],
        'FIX_REACHABILITY': ['Improve support-aware hard negatives, pair features, and balanced reachability diagnostics before expanding experiments.'],
        'FIX_BOUNDARY': ['Switch from direction-only boundary to support/portal-mode overlap and rerun diagnostics-only.'],
        'FIX_GRAPH': ['Profile graph construction; use landmark spectral bottlenecks or cached diagnostic-only reruns before more sweeps.'],
        'FIX_LOW_LEVEL_POLICY': ['Inspect GCBC subgoal execution before planner tuning.'],
        'EXPAND_EVAL': ['Increase eval episodes or expand to large tasks under the same protocol.'],
    }
    return base + extra.get(decision, [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-root', required=True)
    ap.add_argument('--stage', choices=['stage1', 'stage2', 'stage3'], required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--force-collect', action='store_true')
    args = ap.parse_args()

    log_root = Path(args.log_root)
    out_path = Path(args.out or REPORT_DEFAULTS[args.stage])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_dir = log_root / '_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _maybe_collect(log_root, force=args.force_collect)

    numeric_cols = [
        'time_sec', 'reach_auc_proxy', 'reach_auprc_proxy', 'false_positive_proxy_rate', 'cross_traj_selected_rate',
        'reachable_edge_coverage_proxy', 'edge_auc_balanced', 'edge_auprc_balanced', 'supported_edge_rate',
        'selected_supported_rate', 'selected_hard_neg_proxy_rate', 'selected_unlabeled_bridge_rate',
        'psi_mean', 'psi_p10', 'psi_p50', 'psi_p90', 'supported_pair_rate', 'supported_edge_arr_rate', 'supported_edge_dep_rate',
        'found', 'total_risk', 'total_boundary', 'total_cost', 'objective', 'num_edges', 'num_subgoals', 'is_trivial_pair', 'lambda_risk',
        'success', 'return', 'steps', 'replans', 'no_path_count', 'initial_plan_failed_count', 'plan_failed_initial',
        'fallback_used', 'fallback_count', 'direct_goal_attempts', 'last_plan_edges', 'first_plan_edges',
        'max_plan_edges', 'mean_plan_edges', 'num_plan_calls', 'num_subgoal_attempts', 'num_subgoal_reached',
        'subgoal_reach_rate', 'goal_distance_final', 'subgoal_horizon', 'subgoal_threshold', 'success_threshold',
        'lambda_risk', 'lambda_boundary',
        'edge_rollout_auc', 'edge_rollout_auprc', 'success_rate', 'selected_edge_success_rate',
        'unselected_edge_success_rate', 'reset_ok_count', 'reset_unavailable_count', 'reset_available',
        'num_edges_eval', 'num_selected_edges_eval', 'num_unselected_edges_eval',
    ]
    summary_all = _read_csv(analysis_dir / 'summary_all.csv')
    diagnostics_all = _coerce_numeric(_read_csv(analysis_dir / 'diagnostics_all.csv'), numeric_cols)
    graph_all = _coerce_numeric(_read_csv(analysis_dir / 'graph_all.csv'), ['time_sec', 'num_nodes', 'num_edges', 'mean_out_degree', 'p_exec_mean', 'p_exec_p10', 'risk_mean', 'cost_mean', 'duration_sec', 'spectral_seconds', 'spectral_landmark_seconds'])
    eval_all = _coerce_numeric(_read_csv(analysis_dir / 'eval_all.csv'), numeric_cols)
    profile_all = _coerce_numeric(_read_csv(analysis_dir / 'profile_all.csv'), ['duration_sec'])

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
    balanced_edge_diag = _latest_metric_rows(diagnostics_all, 'balanced_edge_diag', ['run_id'])
    boundary_diag = _latest_metric_rows(diagnostics_all, 'boundary_diag', ['run_id'])
    edge_rollout_diag = _latest_metric_rows(diagnostics_all, 'edge_rollout_diag', ['run_id'])
    path_diag = diagnostics_all[diagnostics_all.get('phase', pd.Series(dtype=object)) == 'path_diag'].copy()
    if 'variant' in path_diag.columns:
        path_diag = path_diag[path_diag['variant'].notna()].copy()

    edge_keys = ['env'] if args.stage != 'stage2' else ['env', 'node_method']
    planner_keys = ['env', 'variant'] if args.stage != 'stage2' else ['env', 'node_method', 'variant']
    if 'lambda_risk' in path_diag.columns and path_diag['lambda_risk'].nunique(dropna=True) > 1:
        planner_keys.append('lambda_risk')
    boundary_keys = ['env'] if args.stage != 'stage2' else ['env', 'node_method']

    edge_summary = _round_df(_agg_mean(edge_diag, edge_keys, ['reach_auc_proxy', 'reach_auprc_proxy', 'cross_traj_selected_rate', 'reachable_edge_coverage_proxy', 'selected_edges', 'num_edges']))
    balanced_edge_summary = _round_df(_agg_mean(balanced_edge_diag, edge_keys, ['edge_auc_balanced', 'edge_auprc_balanced', 'supported_edge_rate', 'selected_supported_rate', 'selected_hard_neg_proxy_rate', 'selected_unlabeled_bridge_rate', 'score_supported_mean', 'score_hard_neg_proxy_mean', 'score_unlabeled_bridge_mean']))
    planner_summary_all = _round_df(_planner_summary(path_diag, planner_keys, nonzero_only=False))
    planner_summary_nonzero = _round_df(_planner_summary(path_diag, planner_keys, nonzero_only=True))
    boundary_summary = _round_df(_agg_mean(boundary_diag, boundary_keys, ['psi_mean', 'psi_p10', 'psi_p50', 'psi_p90', 'supported_pair_rate', 'supported_edge_arr_rate', 'supported_edge_dep_rate', 'num_pairs']))
    edge_rollout_summary = _round_df(_agg_mean(edge_rollout_diag, edge_keys, ['edge_rollout_auc', 'edge_rollout_auprc', 'success_rate', 'selected_edge_success_rate', 'unselected_edge_success_rate', 'reset_available', 'reset_ok_count', 'reset_unavailable_count', 'num_edges_eval', 'num_selected_edges_eval', 'num_unselected_edges_eval']))
    eval_keys = ['env', 'variant']
    if 'condition' in eval_all.columns:
        eval_keys = ['condition', 'env', 'variant']
    eval_summary = _round_df(_eval_summary(eval_all, eval_keys))
    graph_summary = _round_df(_agg_mean(graph_all, ['env', 'event'], ['num_nodes', 'num_edges', 'mean_out_degree', 'p_exec_mean', 'risk_mean', 'cost_mean', 'duration_sec', 'spectral_seconds']))
    profile_summary = _round_df(_agg_mean(profile_all, ['env', 'phase', 'event'], ['duration_sec']))

    prefix = args.stage
    for name, df in [
        (f'{prefix}_edge_summary.csv', edge_summary),
        (f'{prefix}_balanced_edge_summary.csv', balanced_edge_summary),
        (f'{prefix}_planner_summary_all_pairs.csv', planner_summary_all),
        (f'{prefix}_planner_summary_nonzero_pairs.csv', planner_summary_nonzero),
        (f'{prefix}_boundary_summary.csv', boundary_summary),
        (f'{prefix}_edge_rollout_summary.csv', edge_rollout_summary),
        (f'{prefix}_graph_summary.csv', graph_summary),
        (f'{prefix}_profile_summary.csv', profile_summary),
    ]:
        if not df.empty:
            df.to_csv(analysis_dir / name, index=False)
    if args.stage == 'stage3' and not eval_summary.empty:
        eval_summary.to_csv(analysis_dir / f'{prefix}_summary.csv', index=False)
    elif not planner_summary_nonzero.empty:
        planner_summary_nonzero.to_csv(analysis_dir / f'{prefix}_summary.csv', index=False)

    failure_modes = _stderr_failure_modes(log_root)
    if args.stage == 'stage1':
        decision = _decision_stage1(run_counts, edge_summary, planner_summary_nonzero, boundary_summary, balanced_edge_summary)
    elif args.stage == 'stage2':
        decision = _decision_stage2(run_counts, planner_summary_nonzero, edge_summary)
    else:
        decision = _decision_stage3(run_counts, eval_summary)

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
        '## Planner Metrics: Nonzero Pairs',
        _markdown_table(planner_summary_nonzero),
        '',
        '## Planner Metrics: All Pairs',
        _markdown_table(planner_summary_all),
        '',
        '## Edge Diagnostics',
        _markdown_table(edge_summary),
        '',
        '## Balanced Edge Diagnostics',
        _markdown_table(balanced_edge_summary),
        '',
        '## Boundary Diagnostics',
        _markdown_table(boundary_summary),
        '',
        '## Edge Rollout Diagnostics',
        _markdown_table(edge_rollout_summary),
        '',
        '## Graph Summary',
        _markdown_table(graph_summary),
        '',
        '## Eval Summary',
        _markdown_table(eval_summary),
        '',
        '## Profile Summary',
        _markdown_table(profile_summary),
        '',
        '## Failure Modes',
        _markdown_table(failure_modes),
        '',
        '## Decision',
        f'- decision: {decision}',
        '',
        '## Next Actions',
    ]
    lines.extend(f'- {item}' for item in _next_actions(args.stage, decision, log_root))
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()

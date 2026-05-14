#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(',') if x.strip()]


def parse_int_csv(s: str) -> list[int]:
    return [int(x) for x in parse_csv(s)]


def task(run_id: str, env: str, seed: int, variant: str, node_method: str = 'bars', set_items: dict | None = None, mem_mb: int | None = None) -> dict:
    out = {'run_id': run_id, 'env': env, 'seed': seed, 'variant': variant, 'node_method': node_method, 'set': set_items or {}}
    if mem_mb is not None:
        out['mem_mb'] = int(mem_mb)
    return out


def write_sweep(path: Path, base_config: str, tasks: list[dict], default_mem_mb: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'base_config': base_config, 'resources': {'default_mem_mb': int(default_mem_mb)}, 'tasks': tasks}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(path)


def build_protocol(envs: Iterable[str], seeds: Iterable[int], episodes: int, warmstart_root: str, default_mem_mb: int) -> list[dict]:
    variants = ['shortest', 'reachability', 'full_bars', 'constrained_bars']
    conditions = {
        'pure_no_fallback': {'eval.fallback_mode': 'none'},
        'after_k_fixed': {'eval.fallback_mode': 'direct_goal_after_k', 'eval.direct_goal_after_k': 3, 'eval.max_deferred_no_path_replans': 5},
        'after_progress': {'eval.fallback_mode': 'direct_goal_after_progress', 'eval.direct_goal_min_subgoals_reached': 1, 'eval.direct_goal_min_distance_improvement': 1.0},
        'after_k_or_progress': {'eval.fallback_mode': 'direct_goal_after_k_or_progress', 'eval.direct_goal_after_k': 3, 'eval.direct_goal_min_subgoals_reached': 1, 'eval.direct_goal_min_distance_improvement': 1.0},
        'direct_goal_fallback_upper': {'eval.fallback_mode': 'direct_goal'},
    }
    tasks: list[dict] = []
    for env in envs:
        for seed in seeds:
            for cond, extra in conditions.items():
                for variant in variants:
                    st = {
                        'eval.enabled': True,
                        'eval.episodes': episodes,
                        'eval.condition': cond,
                        'eval.max_steps': 1000,
                        'eval.subgoal_horizon': 50,
                        'eval.subgoal_threshold': 1.0,
                        'eval.success_threshold': 0.5,
                        'diagnostics.enabled': False,
                        'diagnostics.edge_rollout_enabled': False,
                        'planner.lambda_boundary': 0.1,
                        'planner.exec_budget': 8.0,
                        'planner.max_edges': 12,
                    }
                    st.update(extra)
                    if warmstart_root:
                        st.update({
                            'experiment.warmstart_root': warmstart_root,
                            'experiment.warmstart_source_variant': 'full_bars',
                            'experiment.warmstart_artifacts': ['tdr', 'policy', 'reachability', 'embeddings', 'graph', 'boundary'],
                        })
                    tasks.append(task(f'{cond}_{env}_{variant}_seed{seed}', env, seed, variant, set_items=st, mem_mb=default_mem_mb))
            tasks.append(task(
                f'direct_goal_only_{env}_direct_goal_seed{seed}', env, seed, 'direct_goal', set_items={
                    'eval.enabled': True, 'eval.episodes': episodes, 'eval.condition': 'direct_goal_only',
                    'eval.fallback_mode': 'none', 'eval.max_steps': 1000, 'eval.subgoal_horizon': 50,
                    'eval.subgoal_threshold': 1.0, 'eval.success_threshold': 0.5,
                    'diagnostics.enabled': False, 'diagnostics.edge_rollout_enabled': False,
                }, mem_mb=default_mem_mb
            ))
    return tasks


def build_budget(envs: Iterable[str], seeds: Iterable[int], episodes: int, budgets: Iterable[float], warmstart_root: str, default_mem_mb: int) -> list[dict]:
    tasks: list[dict] = []
    for env in envs:
        for seed in seeds:
            for budget in budgets:
                for fallback in ['none', 'direct_goal_after_progress']:
                    cond = f'budget{budget:g}_{fallback}'
                    st = {
                        'eval.enabled': True,
                        'eval.episodes': episodes,
                        'eval.condition': cond,
                        'eval.fallback_mode': fallback,
                        'eval.max_steps': 1000,
                        'eval.subgoal_horizon': 50,
                        'eval.subgoal_threshold': 1.0,
                        'eval.success_threshold': 0.5,
                        'planner.exec_budget': float(budget),
                        'eval.exec_budget': float(budget),
                        'planner.max_edges': 12,
                        'eval.max_plan_edges': 12,
                        'diagnostics.enabled': False,
                    }
                    if warmstart_root:
                        st.update({
                            'experiment.warmstart_root': warmstart_root,
                            'experiment.warmstart_source_variant': 'full_bars',
                            'experiment.warmstart_artifacts': ['tdr', 'policy', 'reachability', 'embeddings', 'graph', 'boundary'],
                        })
                    tasks.append(task(f'{cond}_{env}_constrained_bars_seed{seed}', env, seed, 'constrained_bars', set_items=st, mem_mb=default_mem_mb))
    return tasks


def build_gas(envs: Iterable[str], seeds: Iterable[int], episodes: int, default_mem_mb: int) -> list[dict]:
    variants = ['shortest', 'reachability', 'full_bars', 'constrained_bars']
    tasks: list[dict] = []
    for env in envs:
        for seed in seeds:
            for variant in variants:
                st = {
                    'external_gas.keygraph_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/keygraph.pkl',
                    'external_gas.dataset_embeddings_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/dataset_embeddings.npy',
                    'external_gas.node_indices_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/node_indices.npy',
                    'embedding.dataset_embeddings_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/dataset_embeddings.npy',
                    'external_policy.repo_path': '${GAS_REPO_PATH}',
                    'external_policy.checkpoint_path': f'${{GAS_POLICY_CKPT_ROOT}}/{env}/seed{seed}/policy.pkl',
                    'external_policy.kwargs.seed': seed,
                    'eval.enabled': True,
                    'eval.episodes': episodes,
                    'eval.condition': 'gas_same_backbone_pure',
                    'eval.fallback_mode': 'none',
                    'eval.embedding_source': 'policy',
                    'eval.max_steps': 1000,
                    'eval.subgoal_horizon': 50,
                    'eval.subgoal_threshold': 1.0,
                    'eval.success_threshold': 0.5,
                    'planner.exec_budget': 8.0,
                    'eval.exec_budget': 8.0,
                    'planner.max_edges': 12,
                    'diagnostics.enabled': False,
                    'graph.load_if_exists': False,
                    'boundary.load_if_exists': False,
                }
                tasks.append(task(f'gas_same_backbone_pure_{env}_{variant}_seed{seed}', env, seed, variant, node_method='bars', set_items=st, mem_mb=default_mem_mb))
            # Recovery-assisted ablation for best BARS planner only.
            st = {
                'external_gas.keygraph_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/keygraph.pkl',
                'external_gas.dataset_embeddings_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/dataset_embeddings.npy',
                'external_gas.node_indices_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/node_indices.npy',
                'embedding.dataset_embeddings_path': f'${{GAS_ARTIFACT_ROOT}}/{env}/seed{seed}/dataset_embeddings.npy',
                'external_policy.repo_path': '${GAS_REPO_PATH}',
                'external_policy.checkpoint_path': f'${{GAS_POLICY_CKPT_ROOT}}/{env}/seed{seed}/policy.pkl',
                'external_policy.kwargs.seed': seed,
                'eval.enabled': True,
                'eval.episodes': episodes,
                'eval.condition': 'gas_same_backbone_after_progress',
                'eval.fallback_mode': 'direct_goal_after_progress',
                'eval.embedding_source': 'policy',
                'eval.max_steps': 1000,
                'eval.subgoal_horizon': 50,
                'eval.subgoal_threshold': 1.0,
                'eval.success_threshold': 0.5,
                'planner.exec_budget': 8.0,
                'eval.exec_budget': 8.0,
                'planner.max_edges': 12,
                'diagnostics.enabled': False,
                'graph.load_if_exists': False,
                'boundary.load_if_exists': False,
            }
            tasks.append(task(f'gas_same_backbone_after_progress_{env}_constrained_bars_seed{seed}', env, seed, 'constrained_bars', node_method='bars', set_items=st, mem_mb=default_mem_mb))
    return tasks


def build_hiql(envs: Iterable[str], seeds: Iterable[int], episodes: int, default_mem_mb: int) -> list[dict]:
    variants = ['shortest', 'reachability', 'full_bars', 'constrained_bars']
    tasks: list[dict] = []
    for env in envs:
        for seed in seeds:
            for variant in variants:
                st = {
                    'external_policy.repo_path': '${HIQL_REPO_PATH}',
                    'external_policy.checkpoint_path': f'${{HIQL_POLICY_CKPT_ROOT}}/{env}/seed{seed}/checkpoint.pkl',
                    'external_policy.kwargs.seed': seed,
                    'eval.enabled': True,
                    'eval.episodes': episodes,
                    'eval.condition': 'hiql_lowlevel_bars_pure',
                    'eval.fallback_mode': 'none',
                    'eval.embedding_source': 'tdr',
                    'eval.max_steps': 1000,
                    'eval.subgoal_horizon': 50,
                    'eval.subgoal_threshold': 1.0,
                    'eval.success_threshold': 0.5,
                    'planner.exec_budget': 8.0,
                    'eval.exec_budget': 8.0,
                    'planner.max_edges': 12,
                    'diagnostics.enabled': False,
                }
                tasks.append(task(f'hiql_lowlevel_bars_pure_{env}_{variant}_seed{seed}', env, seed, variant, node_method='bars', set_items=st, mem_mb=default_mem_mb))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='configs/sweeps')
    ap.add_argument('--envs', default='antmaze-medium-play-v2,antmaze-medium-diverse-v2')
    ap.add_argument('--large-envs', default='antmaze-large-play-v2,antmaze-large-diverse-v2')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--warmstart-root', default='runs_stage16_full12')
    ap.add_argument('--budgets', default='2,4,6,8,10,12')
    ap.add_argument('--default-mem-mb', type=int, default=6000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    envs = parse_csv(args.envs)
    all_envs = envs + [e for e in parse_csv(args.large_envs) if e not in envs]
    seeds = parse_int_csv(args.seeds)
    budgets = [float(x) for x in parse_csv(args.budgets)]

    write_sweep(out / 'd4rl_stage20_protocol_fix_medium100.json', '../routeb/d4rl_antmaze_stage20_bars.json', build_protocol(envs, seeds, args.episodes, args.warmstart_root, args.default_mem_mb), args.default_mem_mb)
    write_sweep(out / 'd4rl_stage20_budget_sweep.json', '../routeb/d4rl_antmaze_stage20_bars.json', build_budget(envs, seeds, args.episodes, budgets, args.warmstart_root, args.default_mem_mb), args.default_mem_mb)
    write_sweep(out / 'd4rl_stage20_routeb_gas_same_backbone.json', '../routeb/d4rl_antmaze_stage20_gas_same_backbone.json', build_gas(all_envs, seeds, args.episodes, args.default_mem_mb), args.default_mem_mb)
    write_sweep(out / 'd4rl_stage20_routeb_hiql_policy.json', '../routeb/d4rl_antmaze_stage20_hiql_policy.json', build_hiql(all_envs, seeds, args.episodes, args.default_mem_mb), args.default_mem_mb)


if __name__ == '__main__':
    main()

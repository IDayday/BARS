from __future__ import annotations

import os
import shutil
import socket
import time
from typing import Dict, Optional
from pathlib import Path

import numpy as np

from bars.common.artifacts import package_logs
from bars.common.checkpoint import load_checkpoint
from bars.common.config import load_json, save_json
from bars.common.device import describe_visible_cuda, get_torch_device
from bars.common.logging import CSVLogger
from bars.common.profile import phase_timer
from bars.common.seed import set_seed
from bars.common.stopper import Stopper
from bars.data.d4rl_dataset import load_d4rl_dataset
from bars.data.toy_dataset import make_toy_dataset
from bars.data.ogbench_dataset import load_ogbench_dataset
from bars.eval.rollout import evaluate_planner_policy
from bars.eval.edge_rollout_diag import run_edge_rollout_diagnostics
from bars.external.gas_compat import convert_gas_keygraph_to_bars_graph
from bars.graph.boundary import BoundaryIndex, build_boundary_index
from bars.graph.diagnostics import run_boundary_diagnostics, run_edge_diagnostics, run_path_diagnostics
from bars.graph.edges import build_edges, _score_reachability
from bars.graph.nodes import select_graph_nodes
from bars.graph.types import BARSGraph
from bars.models.external_policy import build_external_policy_from_config
from bars.models.dataset_embedding import DatasetEmbeddingLookupModel
from bars.models.reachability import ReachabilityModel
from bars.models.policy import GoalConditionedPolicy
from bars.models.bars_iql import BARSIQLPolicy
from bars.training.policy_train import train_policy
from bars.training.reach_train import train_reachability
from bars.training.tdr_train import embed_dataset, train_tdr


def _make_loggers(run_dir: str, cfg: Dict) -> Dict[str, CSVLogger]:
    base = {
        'run_id': cfg.get('run_id', os.path.basename(run_dir)),
        'env': cfg.get('env_name', cfg.get('data', {}).get('env_name', 'unknown')),
        'seed': cfg.get('seed', 0),
        'variant': cfg.get('planner', {}).get('variant', 'full_bars'),
        'node_method': cfg.get('graph', {}).get('node_method', 'bars'),
        'condition': cfg.get('experiment', {}).get('condition', cfg.get('eval', {}).get('protocol_condition', '')),
    }
    return {name: CSVLogger(os.path.join(run_dir, 'logs', f'{name}.csv'), base) for name in ['train', 'graph', 'diagnostics', 'eval', 'summary', 'profile']}


def _load_data(cfg: Dict):
    source = str(cfg.get('data', {}).get('source', 'd4rl')).lower()
    if source == 'toy':
        dc = cfg.get('data', {})
        return None, make_toy_dataset(num_traj=int(dc.get('num_traj', 16)), length=int(dc.get('length', 20)), seed=int(cfg.get('seed', 0)))
    if source == 'd4rl':
        return load_d4rl_dataset(
            cfg.get('data', {}).get('env_name', cfg.get('env_name', 'antmaze-medium-play-v2')),
            dataset_limit=int(cfg.get('data', {}).get('dataset_limit', 0)),
        )
    if source in {'ogbench', 'gas_v0'}:
        dc = cfg.get('data', {})
        return load_ogbench_dataset(
            dc.get('env_name', cfg.get('env_name', 'antmaze-medium-navigate-v0')),
            dataset_limit=int(dc.get('dataset_limit', 0)),
            dataset_path=dc.get('dataset_path', None),
            dataset_dir=dc.get('dataset_dir', None),
            compact_dataset=bool(dc.get('compact_dataset', False)),
        )
    raise ValueError(f'Unknown data.source={source}')


def _expand_path_value(value) -> str:
    if value is None:
        return ""
    return os.path.expandvars(str(value)).strip()


def _is_unresolved_path(value) -> bool:
    expanded = _expand_path_value(value)
    return bool(expanded) and '$' in expanded


def _set_path_unless_clobbering_concrete(target: Dict, key: str, value) -> None:
    if not value:
        return
    current = target.get(key, '')
    if _is_unresolved_path(value) and current and not _is_unresolved_path(current):
        return
    target[key] = value


def _compact_external_embeddings_for_dataset(raw_embeddings: np.ndarray, dataset) -> np.ndarray:
    raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    if len(raw_embeddings) == dataset.size:
        return raw_embeddings
    raw_idx = []
    for sl in dataset.traj_slices:
        raw_idx.extend(range(int(sl.raw_start), max(int(sl.raw_start), int(sl.raw_end) - 1)))
    raw_idx = np.asarray(raw_idx, dtype=np.int64)
    if len(raw_idx) == dataset.size and len(raw_idx) and int(raw_idx.max()) < len(raw_embeddings):
        return raw_embeddings[raw_idx].astype(np.float32)
    raise ValueError(
        f"External dataset embeddings length {len(raw_embeddings)} does not match compact dataset size {dataset.size}; "
        "provide compact embeddings or an export aligned to D4RL raw observations."
    )


def _resolve_external_embedding_path(cfg: Dict) -> str:
    emb_cfg = cfg.get('embedding', {})
    path = _expand_path_value(emb_cfg.get('dataset_embeddings_path', ''))
    unresolved = path if _is_unresolved_path(path) else ''
    if path and not _is_unresolved_path(path):
        return path
    # Exact GAS same-backbone mode commonly stores the official TDR embeddings here.
    path = _expand_path_value(cfg.get('external_gas', {}).get('dataset_embeddings_path', ''))
    if path and bool(emb_cfg.get('auto_use_external_gas_embeddings', True)) and not _is_unresolved_path(path):
        return path
    return unresolved


def _load_or_train_embeddings(cfg: Dict, dataset, run_dir: str, device, train_logger: CSVLogger, summary: CSVLogger, profile_logger: CSVLogger | None, stopper: Optional[Stopper]):
    """Return (tdr_or_lookup_model, embeddings, embedding_path, source_label)."""
    emb_cache = os.path.join(run_dir, 'cache', 'embeddings.npy')
    external_path = _resolve_external_embedding_path(cfg)
    if external_path:
        if not os.path.exists(external_path):
            raise FileNotFoundError(f"Configured external dataset embeddings not found: {external_path}")
        raw = np.load(external_path)
        embeddings = _compact_external_embeddings_for_dataset(raw, dataset).astype(np.float32)
        os.makedirs(os.path.dirname(emb_cache), exist_ok=True)
        np.save(emb_cache, embeddings)
        lookup_dim = int(cfg.get('embedding', {}).get('lookup_obs_dim', cfg.get('eval', {}).get('goal_dim', dataset.obs_dim)))
        model = DatasetEmbeddingLookupModel(dataset.observations, embeddings, raw_obs_dim=lookup_dim)
        train_logger.log({
            'phase': 'tdr',
            'event': 'external_embeddings_loaded',
            'path': external_path,
            'cache_path': emb_cache,
            'latent_dim': int(embeddings.shape[1]),
            'lookup_obs_dim': lookup_dim,
        })
        _summary_log(summary, 'train_tdr_end', 'completed', embedding_path=emb_cache, latent_dim=embeddings.shape[1], embedding_source='external_dataset_embeddings')
        return model, embeddings, emb_cache, 'external_dataset_embeddings'

    _summary_log(summary, 'train_tdr_start', 'running')
    with phase_timer(profile_logger, 'pipeline', 'train_tdr'):
        tdr_model = train_tdr(dataset, cfg, run_dir, device, train_logger, stopper)
    with phase_timer(profile_logger, 'pipeline', 'embed_dataset'):
        embeddings = embed_dataset(
            tdr_model,
            dataset,
            device,
            batch_size=int(cfg.get('tdr', {}).get('embed_batch_size', 8192)),
            cache_path=emb_cache,
            force=bool(cfg.get('tdr', {}).get('force_reembed', False)),
        )
    _summary_log(summary, 'train_tdr_end', 'completed', embedding_path=emb_cache, latent_dim=embeddings.shape[1], embedding_source='bars_tdr')
    return tdr_model, embeddings, emb_cache, 'bars_tdr'


def _summary_log(summary: CSVLogger, phase: str, status: str, **kwargs) -> None:
    summary.log({'phase': phase, 'status': status, **kwargs})


def _apply_routeb_backbone_config(cfg: Dict) -> Dict:
    """Normalize Route-B backbone switches into the existing config structure."""
    cfg = dict(cfg)
    rb = dict(cfg.get('routeb', {}))
    mode = str(rb.get('backbone', rb.get('mode', '')) or '').lower()
    if not mode:
        return cfg
    cfg['routeb'] = rb
    cfg.setdefault('experiment', {})['routeb_backbone'] = mode
    if mode in {'gas_te', 'gas', 'td_aware'}:
        cfg.setdefault('graph', {})['node_method'] = rb.get('node_method', 'gas_te')
        if 'way_steps' in rb:
            cfg['graph']['gas_way_steps'] = rb['way_steps']
        if 'te_threshold' in rb:
            cfg['graph']['gas_te_threshold'] = rb['te_threshold']
    elif mode in {'gas_keygraph', 'gas_official', 'official_gas'}:
        eg = cfg.setdefault('external_gas', {})
        eg['enabled'] = True
        _set_path_unless_clobbering_concrete(eg, 'keygraph_path', rb.get('keygraph_path'))
        _set_path_unless_clobbering_concrete(eg, 'node_indices_path', rb.get('node_indices_path'))
        _set_path_unless_clobbering_concrete(eg, 'dataset_embeddings_path', rb.get('dataset_embeddings_path'))
        if 'rescore_edges' in rb:
            eg['rescore_edges'] = rb['rescore_edges']
        for key in ['bidirectional_edges', 'normalize_cost']:
            if key in rb:
                eg[key] = rb[key]
        _set_path_unless_clobbering_concrete(cfg.setdefault('embedding', {}), 'dataset_embeddings_path', rb.get('dataset_embeddings_path'))
    elif mode in {'hiql', 'hiql_external', 'external_hiql'}:
        cfg.setdefault('policy', {})['type'] = 'external'
        ep = cfg.setdefault('external_policy', {})
        for key in ['repo_path', 'factory', 'factory_path', 'object', 'object_path', 'checkpoint_path', 'act_method']:
            if key in rb:
                ep[key] = rb[key]
        if 'kwargs' in rb:
            ep.setdefault('kwargs', {}).update(rb['kwargs'])
    else:
        cfg.setdefault('experiment', {})['routeb_unknown_backbone'] = mode
    return cfg


def _rescore_graph_edges_with_reachability(graph: BARSGraph, reach_model, cfg: Dict, device, logger: CSVLogger) -> BARSGraph:
    if graph.num_edges == 0 or reach_model is None:
        return graph
    gcfg = cfg.get('graph', {})
    p_exec = _score_reachability(
        reach_model,
        graph.node_embeddings,
        graph.src,
        graph.dst,
        device,
        batch_size=int(gcfg.get('score_batch_size', 32768)),
        fallback_scale=float(np.median(graph.cost) + 1e-6),
    )
    graph.p_exec = p_exec.astype(np.float32)
    graph.risk = (-np.log(np.clip(graph.p_exec, float(gcfg.get('p_clip', 1e-4)), 1.0))).astype(np.float32)
    logger.log({
        'phase': 'graph',
        'event': 'rescored_external_graph',
        'num_edges': graph.num_edges,
        'p_exec_mean': float(np.mean(graph.p_exec)),
        'risk_mean': float(np.mean(graph.risk)),
    })
    return graph


def _stop_requested(stopper: Optional[Stopper], summary: CSVLogger, location: str) -> bool:
    if stopper is None or not stopper.stop_requested:
        return False
    _summary_log(summary, 'terminated', 'terminated', location=location, reason=stopper.reason or 'requested')
    return True


def _stop_after_phase(cfg: Dict, phase: str) -> bool:
    exp_cfg = cfg.get('experiment', {})
    stop_phase = str(exp_cfg.get('stop_after_phase', '') or '').strip().lower()
    return bool(stop_phase) and stop_phase == str(phase).strip().lower()


def _load_reachability_if_available(cfg: Dict, run_dir: str, latent_dim: int, device) -> Optional[ReachabilityModel]:
    ckpt_path = os.path.join(run_dir, 'checkpoints', 'reachability.pt')
    if not os.path.exists(ckpt_path):
        return None
    model = ReachabilityModel(latent_dim, tuple(cfg.get('reachability', {}).get('hidden_dims', [256, 256]))).to(device)
    load_checkpoint(ckpt_path, model, map_location=str(device))
    model.eval()
    return model




def _load_policy_if_available(cfg: Dict, run_dir: str, dataset, device):
    pcfg = cfg.get('policy', {})
    policy_type = str(pcfg.get('type', 'gcbc')).lower()
    if policy_type in {'external', 'hiql_external', 'gas_external'}:
        return build_external_policy_from_config(cfg, dataset, device=device)
    ckpt_path = os.path.join(run_dir, 'checkpoints', 'policy.pt')
    if not os.path.exists(ckpt_path):
        return None
    if policy_type in {'bars_iql', 'bars-low', 'bars_low', 'iql', 'bars_iql_low'}:
        model = BARSIQLPolicy(
            dataset.obs_dim, dataset.action_dim,
            actor_hidden_dims=tuple(pcfg.get('actor_hidden_dims', pcfg.get('hidden_dims', [256, 256, 256]))),
            value_hidden_dims=tuple(pcfg.get('value_hidden_dims', [256, 256])),
            goal_delta=bool(pcfg.get('goal_delta', True)),
            value_temperature=float(pcfg.get('value_temperature', 5.0)),
        ).to(device)
    else:
        model = GoalConditionedPolicy(dataset.obs_dim, dataset.action_dim, tuple(pcfg.get('hidden_dims', [256, 256])), bool(pcfg.get('goal_delta', True))).to(device)
    load_checkpoint(ckpt_path, model, map_location=str(device))
    model.eval()
    return model


def _maybe_materialize_warmstart(cfg: Dict, run_dir: str, summary: CSVLogger) -> None:
    exp_cfg = cfg.get('experiment', {})
    warmstart_root = exp_cfg.get('warmstart_root')
    if not warmstart_root:
        return
    env_name = cfg.get('data', {}).get('env_name', cfg.get('env_name', 'unknown'))
    variant = cfg.get('planner', {}).get('variant', 'full_bars')
    source_variant = exp_cfg.get('warmstart_source_variant', variant)
    seed = int(cfg.get('seed', 0))
    root = Path(str(warmstart_root))
    search_dir = root / str(env_name) / str(source_variant)
    if not search_dir.exists():
        _summary_log(summary, 'warmstart', 'skipped', reason='missing_search_dir', warmstart_root=str(root), env=env_name, variant=variant, source_variant=source_variant, seed=seed)
        return
    candidates = sorted(search_dir.glob(f'*seed{seed}_*'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        _summary_log(summary, 'warmstart', 'skipped', reason='missing_source_run', warmstart_root=str(root), env=env_name, variant=variant, source_variant=source_variant, seed=seed)
        return
    src_run = candidates[-1]
    artifact_map = {
        'tdr': ['checkpoints/tdr.pt'],
        'policy': ['checkpoints/policy.pt'],
        'reachability': ['checkpoints/reachability.pt'],
        'embeddings': ['cache/embeddings.npy'],
        'graph': ['cache/graph.npz'],
        'boundary': ['cache/boundary.npz'],
    }
    requested = exp_cfg.get('warmstart_artifacts', ['tdr', 'policy', 'reachability', 'embeddings'])
    if isinstance(requested, str):
        requested = [x.strip() for x in requested.split(',') if x.strip()]
    requested = [str(x) for x in requested]
    rels = []
    for name in requested:
        rels.extend(artifact_map.get(name, [name] if '/' in name else []))
    copied = []
    skipped_existing = []
    missing = []
    for rel in rels:
        src = src_run / rel
        dst = Path(run_dir) / rel
        if not src.exists():
            missing.append(rel)
            continue
        if dst.exists():
            skipped_existing.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    _summary_log(summary, 'warmstart', 'completed', source_run_dir=str(src_run), source_variant=source_variant, requested=';'.join(requested), copied_count=len(copied), copied=';'.join(copied), missing=';'.join(missing), skipped_existing=';'.join(skipped_existing))

def run_cached_diagnostics(cfg: Dict, run_dir: str, stopper: Optional[Stopper] = None) -> str:
    """Rerun only diagnostics from cached embeddings/graph/boundary/checkpoint.

    This is intended for Stage 1.5. It makes lambda/path/edge diagnostic sweeps
    cheap instead of rebuilding the expensive graph for every analysis change.
    """
    cfg = _apply_routeb_backbone_config(cfg)
    cfg.setdefault('run_id', os.path.basename(run_dir))
    cfg.setdefault('env_name', cfg.get('data', {}).get('env_name', 'unknown'))
    loggers = _make_loggers(run_dir, cfg)
    summary = loggers['summary']
    start = time.time()
    status = 'started'
    try:
        set_seed(int(cfg.get('seed', 0)))
        device = get_torch_device(str(cfg.get('device', 'cuda')))
        _summary_log(summary, 'diagnostics_only_start', 'running', run_dir=run_dir)
        _summary_log(summary, 'load_dataset', 'running')
        _, dataset = _load_data(cfg)
        _summary_log(summary, 'load_dataset', 'completed', dataset_size=dataset.size, num_trajectories=dataset.num_trajectories)
        emb_path = os.path.join(run_dir, 'cache', 'embeddings.npy')
        graph_path = os.path.join(run_dir, 'cache', 'graph.npz')
        boundary_path = os.path.join(run_dir, 'cache', 'boundary.npz')
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f'Missing embeddings cache: {emb_path}')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f'Missing graph cache: {graph_path}')
        embeddings = np.load(emb_path)
        graph = BARSGraph.load_npz(graph_path)
        boundary = BoundaryIndex.load_npz(boundary_path) if os.path.exists(boundary_path) and bool(cfg.get('boundary', {}).get('enabled', True)) else None
        reach_model = _load_reachability_if_available(cfg, run_dir, embeddings.shape[1], device)
        _summary_log(summary, 'diagnostics_start', 'running', diagnostics_only=1, reachability_loaded=int(reach_model is not None), boundary_loaded=int(boundary is not None))
        run_edge_diagnostics(dataset, embeddings, graph, cfg, loggers['diagnostics'])
        run_boundary_diagnostics(graph, boundary, cfg, loggers['diagnostics'])
        run_path_diagnostics(dataset, embeddings, graph, boundary, cfg, loggers['diagnostics'])
        _summary_log(summary, 'diagnostics_end', 'completed', diagnostics_only=1)
        status = 'completed' if not (stopper and stopper.stop_requested) else 'terminated'
        return run_dir
    except Exception as exc:
        status = 'failed'
        _summary_log(summary, 'failed', 'failed', error=repr(exc), diagnostics_only=1)
        raise
    finally:
        elapsed = time.time() - start
        try:
            _summary_log(summary, 'package_start', 'running', elapsed_sec=elapsed)
            save_json({'run_id': cfg.get('run_id', os.path.basename(run_dir)), 'status': status, 'elapsed_sec': elapsed, 'run_dir': run_dir, 'diagnostics_only': True, 'created_at': time.time()}, os.path.join(run_dir, 'manifest.json'))
            archive = package_logs(run_dir)
            _summary_log(summary, 'package_end', 'completed', archive_path=archive, elapsed_sec=elapsed)
            _summary_log(summary, status, status, elapsed_sec=elapsed, archive_path=archive, diagnostics_only=1)
        except Exception:
            pass


def run_experiment(cfg: Dict, run_dir: str, stopper: Optional[Stopper] = None) -> str:
    os.makedirs(run_dir, exist_ok=True)
    cfg = _apply_routeb_backbone_config(cfg)
    cfg.setdefault('run_id', os.path.basename(run_dir))
    cfg.setdefault('env_name', cfg.get('data', {}).get('env_name', 'unknown'))
    save_json(cfg, os.path.join(run_dir, 'config.json'))
    loggers = _make_loggers(run_dir, cfg)
    summary = loggers['summary']
    start = time.time()
    status = 'started'
    try:
        set_seed(int(cfg.get('seed', 0)))
        device = get_torch_device(str(cfg.get('device', 'cuda')))
        _maybe_materialize_warmstart(cfg, run_dir, summary)
        _summary_log(summary, 'start', 'started', host=socket.gethostname(), cuda=describe_visible_cuda(), run_dir=run_dir)

        _summary_log(summary, 'load_dataset', 'running')
        with phase_timer(loggers.get('profile'), 'pipeline', 'load_dataset'):
            env, dataset = _load_data(cfg)
        _summary_log(summary, 'load_dataset', 'completed', dataset_size=dataset.size, num_trajectories=dataset.num_trajectories, obs_dim=dataset.obs_dim, action_dim=dataset.action_dim)
        if _stop_requested(stopper, summary, 'after_load_dataset'):
            status = 'terminated'
            return run_dir

        tdr_model, embeddings, emb_cache, embedding_source = _load_or_train_embeddings(
            cfg, dataset, run_dir, device, loggers['train'], summary, loggers.get('profile'), stopper
        )
        if _stop_requested(stopper, summary, 'after_train_tdr'):
            status = 'terminated'
            return run_dir

        _summary_log(summary, 'train_policy_start', 'running')
        with phase_timer(loggers.get('profile'), 'pipeline', 'train_policy'):
            policy = train_policy(dataset, cfg, run_dir, device, loggers['train'], stopper, embeddings=embeddings)
        _summary_log(summary, 'train_policy_end', 'completed')
        if _stop_requested(stopper, summary, 'after_train_policy'):
            status = 'terminated'
            return run_dir

        _summary_log(summary, 'train_reachability_start', 'running', enabled=int(bool(cfg.get('reachability', {}).get('enabled', True))))
        with phase_timer(loggers.get('profile'), 'pipeline', 'train_reachability'):
            reach_model = train_reachability(dataset, embeddings, cfg, run_dir, device, loggers['train'], stopper, policy=policy) if bool(cfg.get('reachability', {}).get('enabled', True)) else None
        _summary_log(summary, 'train_reachability_end', 'completed', enabled=int(reach_model is not None))
        if _stop_requested(stopper, summary, 'after_train_reachability'):
            status = 'terminated'
            return run_dir

        graph_path = os.path.join(run_dir, 'cache', 'graph.npz')
        _summary_log(summary, 'graph_build_start', 'running', graph_path=graph_path)
        if bool(cfg.get('graph', {}).get('load_if_exists', True)) and os.path.exists(graph_path):
            graph = BARSGraph.load_npz(graph_path)
            loggers['graph'].log({'phase': 'graph', 'event': 'loaded', 'path': graph_path, 'num_nodes': graph.num_nodes, 'num_edges': graph.num_edges})
        else:
            external_gas_cfg = cfg.get('external_gas', {})
            gas_keygraph_path = str(external_gas_cfg.get('keygraph_path', '') or '').strip()
            if bool(external_gas_cfg.get('enabled', False)) and gas_keygraph_path and os.path.exists(gas_keygraph_path):
                with phase_timer(loggers.get('profile'), 'graph_build', 'import_gas_keygraph'):
                    graph = convert_gas_keygraph_to_bars_graph(gas_keygraph_path, embeddings, cfg, loggers['graph'], dataset=dataset)
                if bool(external_gas_cfg.get('rescore_edges', True)):
                    with phase_timer(loggers.get('profile'), 'graph_build', 'rescore_external_graph'):
                        graph = _rescore_graph_edges_with_reachability(graph, reach_model, cfg, device, loggers['graph'])
            else:
                with phase_timer(loggers.get('profile'), 'graph_build', 'select_nodes'):
                    node_indices = select_graph_nodes(dataset, embeddings, cfg, loggers['graph'])
                with phase_timer(loggers.get('profile'), 'graph_build', 'build_edges'):
                    graph = build_edges(dataset, embeddings, node_indices, reach_model, cfg, device, loggers['graph'])
            with phase_timer(loggers.get('profile'), 'graph_build', 'save_graph'):
                graph.save_npz(graph_path)
            loggers['graph'].log({'phase': 'graph', 'event': 'saved', 'path': graph_path, 'num_nodes': graph.num_nodes, 'num_edges': graph.num_edges})
        _summary_log(summary, 'graph_build_end', 'completed', graph_path=graph_path, num_nodes=graph.num_nodes, num_edges=graph.num_edges)
        if _stop_requested(stopper, summary, 'after_graph_build'):
            status = 'terminated'
            return run_dir

        boundary = None
        boundary_path = os.path.join(run_dir, 'cache', 'boundary.npz')
        if bool(cfg.get('boundary', {}).get('enabled', True)):
            if bool(cfg.get('boundary', {}).get('load_if_exists', True)) and os.path.exists(boundary_path):
                boundary = BoundaryIndex.load_npz(boundary_path)
                loggers['graph'].log({'phase': 'boundary', 'event': 'loaded', 'path': boundary_path})
            else:
                with phase_timer(loggers.get('profile'), 'graph_build', 'build_boundary'):
                    boundary = build_boundary_index(dataset, embeddings, graph, cfg, loggers['graph'])
                with phase_timer(loggers.get('profile'), 'graph_build', 'save_boundary'):
                    boundary.save_npz(boundary_path)
                loggers['graph'].log({'phase': 'boundary', 'event': 'saved', 'path': boundary_path})
        if _stop_after_phase(cfg, 'graph_build'):
            status = 'completed_graph_timing'
            _summary_log(summary, 'graph_timing_stop', 'completed_graph_timing', location='after_boundary_build', boundary_enabled=int(bool(cfg.get('boundary', {}).get('enabled', True))))
            return run_dir

        _summary_log(summary, 'diagnostics_start', 'running', enabled=int(bool(cfg.get('diagnostics', {}).get('enabled', True))))
        if bool(cfg.get('diagnostics', {}).get('enabled', True)):
            with phase_timer(loggers.get('profile'), 'diagnostics', 'edge'):
                run_edge_diagnostics(dataset, embeddings, graph, cfg, loggers['diagnostics'])
            with phase_timer(loggers.get('profile'), 'diagnostics', 'boundary'):
                run_boundary_diagnostics(graph, boundary, cfg, loggers['diagnostics'])
            with phase_timer(loggers.get('profile'), 'diagnostics', 'path'):
                run_path_diagnostics(dataset, embeddings, graph, boundary, cfg, loggers['diagnostics'])
            if env is not None and bool(cfg.get('diagnostics', {}).get('edge_rollout_enabled', cfg.get('diagnostics', {}).get('edge_rollout', {}).get('enabled', False) if isinstance(cfg.get('diagnostics', {}).get('edge_rollout', {}), dict) else False)):
                with phase_timer(loggers.get('profile'), 'diagnostics', 'edge_rollout'):
                    run_edge_rollout_diagnostics(env, dataset, policy, graph, cfg, device, loggers['diagnostics'], stopper)
        _summary_log(summary, 'diagnostics_end', 'completed', enabled=int(bool(cfg.get('diagnostics', {}).get('enabled', True))))
        if _stop_requested(stopper, summary, 'after_diagnostics'):
            status = 'terminated'
            return run_dir

        _summary_log(summary, 'eval_start', 'running', enabled=int(env is not None and bool(cfg.get('eval', {}).get('enabled', False))))
        if env is not None:
            evaluate_planner_policy(env, dataset, tdr_model, policy, graph, boundary, cfg, device, loggers['eval'], stopper)
        _summary_log(summary, 'eval_end', 'completed', enabled=int(env is not None and bool(cfg.get('eval', {}).get('enabled', False))))
        status = 'completed' if not (stopper and stopper.stop_requested) else 'terminated'
        return run_dir
    except Exception as exc:
        status = 'failed'
        _summary_log(summary, 'failed', 'failed', error=repr(exc))
        raise
    finally:
        elapsed = time.time() - start
        try:
            _summary_log(summary, 'package_start', 'running', elapsed_sec=elapsed)
            save_json({'run_id': cfg.get('run_id', os.path.basename(run_dir)), 'status': status, 'elapsed_sec': elapsed, 'run_dir': run_dir, 'created_at': time.time()}, os.path.join(run_dir, 'manifest.json'))
            archive = package_logs(run_dir)
            _summary_log(summary, 'package_end', 'completed', archive_path=archive, elapsed_sec=elapsed)
            _summary_log(summary, status, status, elapsed_sec=elapsed, archive_path=archive)
        except Exception:
            pass



def rerun_diagnostics(cfg: Dict, run_dir: str, clear: bool = False, rebuild_boundary: bool = False, package: bool = False) -> str:
    """Rerun only graph diagnostics from cached embeddings/graph/boundary.

    This is the Stage-1.5/Stage-2a workhorse: it lets us change diagnostic
    sampling, path lambda sweeps, or boundary scoring without retraining TDR,
    policy, reachability, or rebuilding graph edges.
    """
    cfg = _apply_routeb_backbone_config(cfg)
    cfg.setdefault('run_id', os.path.basename(run_dir))
    cfg.setdefault('env_name', cfg.get('data', {}).get('env_name', 'unknown'))
    loggers = _make_loggers(run_dir, cfg)
    summary = loggers['summary']
    if clear:
        diag_path = os.path.join(run_dir, 'logs', 'diagnostics.csv')
        if os.path.exists(diag_path):
            os.rename(diag_path, diag_path + f'.bak_{int(time.time())}')
        loggers = _make_loggers(run_dir, cfg)
        summary = loggers['summary']
    _summary_log(summary, 'diagnostics_only_start', 'running', clear=int(clear), rebuild_boundary=int(rebuild_boundary))
    env, dataset = _load_data(cfg)
    emb_cache = os.path.join(run_dir, 'cache', 'embeddings.npy')
    graph_path = os.path.join(run_dir, 'cache', 'graph.npz')
    boundary_path = os.path.join(run_dir, 'cache', 'boundary.npz')
    if not os.path.exists(emb_cache):
        raise FileNotFoundError(f'Missing cached embeddings: {emb_cache}')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f'Missing cached graph: {graph_path}')
    embeddings = np.load(emb_cache)
    graph = BARSGraph.load_npz(graph_path)
    boundary = None
    if bool(cfg.get('boundary', {}).get('enabled', True)):
        if os.path.exists(boundary_path) and not rebuild_boundary:
            boundary = BoundaryIndex.load_npz(boundary_path)
            loggers['graph'].log({'phase': 'boundary', 'event': 'loaded_for_diagnostics_only', 'path': boundary_path, 'method': boundary.method})
        else:
            boundary = build_boundary_index(dataset, embeddings, graph, cfg, loggers['graph'])
            boundary.save_npz(boundary_path)
            loggers['graph'].log({'phase': 'boundary', 'event': 'saved_for_diagnostics_only', 'path': boundary_path, 'method': boundary.method})
    device = get_torch_device(str(cfg.get('device', 'cuda')))
    reach_model = _load_reachability_if_available(cfg, run_dir, embeddings.shape[1], device)
    policy = _load_policy_if_available(cfg, run_dir, dataset, device)
    run_edge_diagnostics(dataset, embeddings, graph, cfg, loggers['diagnostics'])
    run_boundary_diagnostics(graph, boundary, cfg, loggers['diagnostics'])
    run_path_diagnostics(dataset, embeddings, graph, boundary, cfg, loggers['diagnostics'])
    dcfg = cfg.get('diagnostics', {})
    edge_rollout_enabled = bool(dcfg.get('edge_rollout_enabled', dcfg.get('edge_rollout', {}).get('enabled', False) if isinstance(dcfg.get('edge_rollout', {}), dict) else False))
    if env is not None and edge_rollout_enabled:
        if policy is None:
            loggers['diagnostics'].log({'phase': 'edge_rollout_diag', 'event': 'completed', 'enabled': 1, 'available': 0, 'reason': 'missing_policy_checkpoint'})
        else:
            run_edge_rollout_diagnostics(env, dataset, policy, graph, cfg, device, loggers['diagnostics'])
    _summary_log(summary, 'diagnostics_only_end', 'completed')
    if package:
        archive = package_logs(run_dir)
        _summary_log(summary, 'diagnostics_only_packaged', 'completed', archive_path=archive)
    try:
        if env is not None:
            env.close()
    except Exception:
        pass
    return run_dir

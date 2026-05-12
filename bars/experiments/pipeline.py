from __future__ import annotations

import os
import socket
import time
from typing import Dict, Optional

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
from bars.eval.rollout import evaluate_planner_policy
from bars.eval.edge_rollout_diag import run_edge_rollout_diagnostics
from bars.graph.boundary import BoundaryIndex, build_boundary_index
from bars.graph.diagnostics import run_boundary_diagnostics, run_edge_diagnostics, run_path_diagnostics
from bars.graph.edges import build_edges
from bars.graph.nodes import select_graph_nodes
from bars.graph.types import BARSGraph
from bars.models.reachability import ReachabilityModel
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
    raise ValueError(f'Unknown data.source={source}')


def _summary_log(summary: CSVLogger, phase: str, status: str, **kwargs) -> None:
    summary.log({'phase': phase, 'status': status, **kwargs})


def _stop_requested(stopper: Optional[Stopper], summary: CSVLogger, location: str) -> bool:
    if stopper is None or not stopper.stop_requested:
        return False
    _summary_log(summary, 'terminated', 'terminated', location=location, reason=stopper.reason or 'requested')
    return True


def _load_reachability_if_available(cfg: Dict, run_dir: str, latent_dim: int, device) -> Optional[ReachabilityModel]:
    ckpt_path = os.path.join(run_dir, 'checkpoints', 'reachability.pt')
    if not os.path.exists(ckpt_path):
        return None
    model = ReachabilityModel(latent_dim, tuple(cfg.get('reachability', {}).get('hidden_dims', [256, 256]))).to(device)
    load_checkpoint(ckpt_path, model, map_location=str(device))
    model.eval()
    return model


def run_cached_diagnostics(cfg: Dict, run_dir: str, stopper: Optional[Stopper] = None) -> str:
    """Rerun only diagnostics from cached embeddings/graph/boundary/checkpoint.

    This is intended for Stage 1.5. It makes lambda/path/edge diagnostic sweeps
    cheap instead of rebuilding the expensive graph for every analysis change.
    """
    cfg = dict(cfg)
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
    cfg = dict(cfg)
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
        _summary_log(summary, 'start', 'started', host=socket.gethostname(), cuda=describe_visible_cuda(), run_dir=run_dir)

        _summary_log(summary, 'load_dataset', 'running')
        with phase_timer(loggers.get('profile'), 'pipeline', 'load_dataset'):
            env, dataset = _load_data(cfg)
        _summary_log(summary, 'load_dataset', 'completed', dataset_size=dataset.size, num_trajectories=dataset.num_trajectories, obs_dim=dataset.obs_dim, action_dim=dataset.action_dim)
        if _stop_requested(stopper, summary, 'after_load_dataset'):
            status = 'terminated'
            return run_dir

        _summary_log(summary, 'train_tdr_start', 'running')
        with phase_timer(loggers.get('profile'), 'pipeline', 'train_tdr'):
            tdr_model = train_tdr(dataset, cfg, run_dir, device, loggers['train'], stopper)
        emb_cache = os.path.join(run_dir, 'cache', 'embeddings.npy')
        with phase_timer(loggers.get('profile'), 'pipeline', 'embed_dataset'):
            embeddings = embed_dataset(
                tdr_model,
                dataset,
                device,
                batch_size=int(cfg.get('tdr', {}).get('embed_batch_size', 8192)),
                cache_path=emb_cache,
                force=bool(cfg.get('tdr', {}).get('force_reembed', False)),
            )
        _summary_log(summary, 'train_tdr_end', 'completed', embedding_path=emb_cache, latent_dim=embeddings.shape[1])
        if _stop_requested(stopper, summary, 'after_train_tdr'):
            status = 'terminated'
            return run_dir

        _summary_log(summary, 'train_policy_start', 'running')
        with phase_timer(loggers.get('profile'), 'pipeline', 'train_policy'):
            policy = train_policy(dataset, cfg, run_dir, device, loggers['train'], stopper)
        _summary_log(summary, 'train_policy_end', 'completed')
        if _stop_requested(stopper, summary, 'after_train_policy'):
            status = 'terminated'
            return run_dir

        _summary_log(summary, 'train_reachability_start', 'running', enabled=int(bool(cfg.get('reachability', {}).get('enabled', True))))
        with phase_timer(loggers.get('profile'), 'pipeline', 'train_reachability'):
            reach_model = train_reachability(dataset, embeddings, cfg, run_dir, device, loggers['train'], stopper) if bool(cfg.get('reachability', {}).get('enabled', True)) else None
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
    cfg = dict(cfg)
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
    run_edge_diagnostics(dataset, embeddings, graph, cfg, loggers['diagnostics'])
    run_boundary_diagnostics(graph, boundary, cfg, loggers['diagnostics'])
    run_path_diagnostics(dataset, embeddings, graph, boundary, cfg, loggers['diagnostics'])
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

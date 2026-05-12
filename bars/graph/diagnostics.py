from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from .boundary import BoundaryIndex
from .planner import plan_path
from .support import sample_edge_support_counts
from .types import BARSGraph
from .ann import KNNIndex


def safe_auc(labels: np.ndarray, scores: np.ndarray):
    labels = labels.astype(np.int32)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float('nan'), float('nan')
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _quantiles(prefix: str, x: np.ndarray) -> Dict[str, float]:
    if len(x) == 0:
        return {f'{prefix}_mean': float('nan'), f'{prefix}_std': float('nan'), f'{prefix}_p10': float('nan'), f'{prefix}_p50': float('nan'), f'{prefix}_p90': float('nan')}
    return {
        f'{prefix}_mean': float(np.mean(x)),
        f'{prefix}_std': float(np.std(x)),
        f'{prefix}_p10': float(np.quantile(x, 0.10)),
        f'{prefix}_p50': float(np.quantile(x, 0.50)),
        f'{prefix}_p90': float(np.quantile(x, 0.90)),
    }


def run_edge_diagnostics(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> None:
    logger.log({'phase': 'edge_diag', 'event': 'start', 'num_edges': graph.num_edges})
    dcfg = cfg.get('diagnostics', {})
    horizon = int(dcfg.get('edge_label_horizon', cfg.get('reachability', {}).get('horizon', 30)))
    gi_src = graph.node_indices[graph.src]
    gi_dst = graph.node_indices[graph.dst]
    same = dataset.traj_id[gi_src] == dataset.traj_id[gi_dst]
    dt = dataset.timestep[gi_dst] - dataset.timestep[gi_src]
    pos_exact = same & (dt > 0) & (dt <= horizon)
    cross = ~same
    valid_exact = pos_exact | cross
    auc_exact, auprc_exact = safe_auc(pos_exact[valid_exact].astype(np.int32), graph.p_exec[valid_exact])
    selected = graph.p_exec >= float(dcfg.get('edge_selected_threshold', 0.5))
    cross_selected_rate = float((selected & cross).sum() / max(1, selected.sum()))

    row = {
        'phase': 'edge_diag',
        'event': 'exact_proxy_completed',
        'edge_label_horizon': horizon,
        'num_edges': graph.num_edges,
        'num_pos_proxy': int(pos_exact.sum()),
        'num_cross_traj_proxy': int(cross.sum()),
        'reach_auc_proxy': auc_exact,
        'reach_auprc_proxy': auprc_exact,
        'selected_threshold': float(dcfg.get('edge_selected_threshold', 0.5)),
        'selected_edges': int(selected.sum()),
        # Kept for backward compatibility, but it should not be interpreted as
        # true false positives because cross-trajectory edges can be valid bridges.
        'false_positive_proxy_rate': cross_selected_rate,
        'cross_traj_selected_rate': cross_selected_rate,
        'reachable_edge_coverage_proxy': float((selected & pos_exact).sum() / max(1, pos_exact.sum())),
    }
    logger.log(row)

    # Balanced support diagnostic: positive labels are graph edges actually hit
    # by sampled same-trajectory segments after nearest-node projection. Edges
    # without observed support are hard-negative proxies, not true negatives.
    if bool(dcfg.get('balanced_edge_diag', True)):
        rng = np.random.default_rng(int(cfg.get('seed', 0)) + 811)
        support = sample_edge_support_counts(
            dataset,
            embeddings,
            graph,
            horizon=horizon,
            num_segments=int(dcfg.get('edge_support_segments', 200000)),
            min_dt=int(dcfg.get('edge_support_min_dt', 1)),
            rng=rng,
            batch_size=int(dcfg.get('edge_support_batch_size', 65536)),
            logger=logger,
            phase='edge_support_diag',
            cfg=cfg,
        )
        supported = support.counts > 0
        unsupported = ~supported
        # Subsample unsupported edges for a balanced AUC/AUPRC proxy. This avoids
        # the misleading extreme class imbalance of exact graph-node labels.
        pos_ids = np.where(supported)[0]
        neg_pool = np.where(unsupported)[0]
        neg_ids = neg_pool
        if len(pos_ids) > 0 and len(neg_pool) > len(pos_ids):
            neg_ids = rng.choice(neg_pool, size=len(pos_ids), replace=False)
        labels = np.concatenate([np.ones(len(pos_ids), dtype=np.int32), np.zeros(len(neg_ids), dtype=np.int32)])
        scores = np.concatenate([graph.p_exec[pos_ids], graph.p_exec[neg_ids]]) if len(labels) else np.empty(0, dtype=np.float32)
        auc_bal, auprc_bal = safe_auc(labels, scores)
        hard_neg_proxy = unsupported & cross
        unlabeled = cross & supported
        out = {
            'phase': 'balanced_edge_diag',
            'event': 'completed',
            'edge_label_horizon': horizon,
            'support_sampled_segments': support.sampled_segments,
            'support_edge_hits': support.edge_hits,
            'support_hit_rate': support.hit_rate,
            'num_supported_edges': int(supported.sum()),
            'supported_edge_rate': float(supported.mean()) if len(supported) else 0.0,
            'num_hard_neg_proxy_edges': int(hard_neg_proxy.sum()),
            'num_unlabeled_bridge_edges': int(unlabeled.sum()),
            'edge_auc_balanced': auc_bal,
            'edge_auprc_balanced': auprc_bal,
            'selected_supported_rate': float((selected & supported).sum() / max(1, supported.sum())),
            'selected_hard_neg_proxy_rate': float((selected & hard_neg_proxy).sum() / max(1, hard_neg_proxy.sum())),
            'selected_unlabeled_bridge_rate': float((selected & unlabeled).sum() / max(1, unlabeled.sum())),
        }
        out.update(_quantiles('score_supported', graph.p_exec[supported]))
        out.update(_quantiles('score_hard_neg_proxy', graph.p_exec[hard_neg_proxy]))
        out.update(_quantiles('score_unlabeled_bridge', graph.p_exec[unlabeled]))
        logger.log(out)


def run_boundary_diagnostics(graph: BARSGraph, boundary: Optional[BoundaryIndex], cfg: Dict, logger: CSVLogger) -> None:
    logger.log({'phase': 'boundary_diag', 'event': 'start', 'enabled': int(boundary is not None)})
    if boundary is None:
        logger.log({'phase': 'boundary_diag', 'event': 'completed', 'enabled': 0})
        return
    psis = []
    supported = []
    out = graph.outgoing_edges()
    for eid in range(graph.num_edges):
        for ne in out[int(graph.dst[eid])]:
            ne = int(ne)
            psis.append(boundary.psi(eid, ne))
            supported.append(int(boundary.has_arr[eid] and boundary.has_dep[ne]))
    if not psis:
        logger.log({'phase': 'boundary_diag', 'event': 'completed', 'enabled': 1, 'num_pairs': 0, 'boundary_method': boundary.method})
        return
    p = np.asarray(psis, dtype=np.float32)
    logger.log({
        'phase': 'boundary_diag',
        'event': 'completed',
        'enabled': 1,
        'boundary_method': boundary.method,
        'num_pairs': len(p),
        'psi_mean': float(p.mean()),
        'psi_p10': float(np.quantile(p, 0.10)),
        'psi_p50': float(np.quantile(p, 0.50)),
        'psi_p90': float(np.quantile(p, 0.90)),
        'supported_pair_rate': float(np.mean(supported)),
        'supported_edge_arr_rate': float(np.mean(boundary.has_arr)) if len(boundary.has_arr) else 0.0,
        'supported_edge_dep_rate': float(np.mean(boundary.has_dep)) if len(boundary.has_dep) else 0.0,
    })


def run_path_diagnostics(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, boundary: Optional[BoundaryIndex], cfg: Dict, logger: CSVLogger) -> None:
    dcfg = cfg.get('diagnostics', {})
    variants = dcfg.get('planner_variants', ['shortest', 'reachability', 'full_bars'])
    num_pairs = int(dcfg.get('num_path_pairs', 256))
    min_dt = int(dcfg.get('path_min_dt', cfg.get('reachability', {}).get('horizon', 30) * 2))
    max_dt = int(dcfg.get('path_max_dt', 250))
    path_min_graph_edges = int(dcfg.get('path_min_graph_edges', 1))
    include_trivial = bool(dcfg.get('include_trivial_path_pairs', False))
    max_attempts = int(dcfg.get('max_sampling_attempts', max(1000, num_pairs * 50)))
    lambda_values = dcfg.get('lambda_risk_values', [cfg.get('planner', {}).get('lambda_risk', 1.0)])
    lambda_values = [float(x) for x in lambda_values]
    lambda_b = float(cfg.get('planner', {}).get('lambda_boundary', 1.0))
    rng = np.random.default_rng(int(cfg.get('seed', 0)) + 131)
    nbrs = KNNIndex.from_config(graph.node_embeddings, cfg, prefix="ann")
    logger.log({
        'phase': 'path_diag',
        'event': 'start',
        'num_pairs': num_pairs,
        'variants': variants,
        'path_min_graph_edges': path_min_graph_edges,
        'include_trivial_path_pairs': int(include_trivial),
        'lambda_risk_values': lambda_values,
    })
    accepted = 0
    attempts = 0
    skipped_same_node = 0
    skipped_short_graph = 0
    progress_every = max(1, num_pairs // 4)
    while accepted < num_pairs and attempts < max_attempts:
        attempts += 1
        try:
            i, j, dt = dataset.sample_future_pairs(1, max_dt, rng, min_dt=min_dt)
        except Exception:
            break
        i = int(i[0])
        j = int(j[0])
        dt = int(dt[0])
        s_node = int(nbrs.kneighbors(embeddings[i:i + 1], 1, return_distance=False)[0, 0])
        g_node = int(nbrs.kneighbors(embeddings[j:j + 1], 1, return_distance=False)[0, 0])
        if s_node == g_node and not include_trivial:
            skipped_same_node += 1
            continue

        # Use shortest as a cheap graph-distance prefilter. This prevents long-horizon
        # diagnostics from being dominated by zero-edge or one-node pairs.
        pre = plan_path(graph, s_node, g_node, variant='shortest', lambda_risk=0.0, lambda_boundary=lambda_b, boundary=None)
        if pre.found and len(pre.edge_path) < path_min_graph_edges and not include_trivial:
            skipped_short_graph += 1
            continue

        pair_id = accepted
        for lambda_r in lambda_values:
            for variant in variants:
                result = plan_path(graph, s_node, g_node, variant=variant, lambda_risk=lambda_r, lambda_boundary=lambda_b, boundary=boundary)
                row = {
                    'phase': 'path_diag',
                    'pair_id': pair_id,
                    'attempt_id': attempts,
                    'start_index': i,
                    'goal_index': j,
                    'true_future_dt': dt,
                    'start_node': s_node,
                    'goal_node': g_node,
                    'is_trivial_pair': int(s_node == g_node),
                    'lambda_risk': float(lambda_r),
                    'lambda_boundary': float(lambda_b),
                    **result.to_row(),
                }
                logger.log(row)
        accepted += 1
        if (accepted % progress_every) == 0 or accepted == num_pairs:
            logger.log({
                'phase': 'path_diag',
                'event': 'progress',
                'pairs_done': accepted,
                'num_pairs': num_pairs,
                'attempts': attempts,
                'skipped_same_node': skipped_same_node,
                'skipped_short_graph': skipped_short_graph,
            })
    logger.log({
        'phase': 'path_diag',
        'event': 'completed',
        'num_pairs': num_pairs,
        'accepted_pairs': accepted,
        'attempts': attempts,
        'skipped_same_node': skipped_same_node,
        'skipped_short_graph': skipped_short_graph,
        'zero_edge_rate_attempt_proxy': float(skipped_same_node / max(1, attempts)),
        'variants': variants,
        'lambda_risk_values': lambda_values,
    })

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from .boundary import BoundaryIndex
from .planner import plan_path
from .types import BARSGraph

def safe_auc(labels: np.ndarray, scores: np.ndarray):
    labels=labels.astype(np.int32)
    if len(np.unique(labels))<2: return float('nan'), float('nan')
    return float(roc_auc_score(labels,scores)), float(average_precision_score(labels,scores))
def run_edge_diagnostics(dataset: OfflineDataset, graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> None:
    dcfg=cfg.get('diagnostics',{}); horizon=int(dcfg.get('edge_label_horizon',cfg.get('reachability',{}).get('horizon',30))); gi_src=graph.node_indices[graph.src]; gi_dst=graph.node_indices[graph.dst]; same=dataset.traj_id[gi_src]==dataset.traj_id[gi_dst]; dt=dataset.timestep[gi_dst]-dataset.timestep[gi_src]; pos=same & (dt>0) & (dt<=horizon); neg=~same; valid=pos|neg; auc,auprc=safe_auc(pos[valid].astype(np.int32),graph.p_exec[valid]); selected=graph.p_exec>=float(dcfg.get('edge_selected_threshold',0.5)); fp=float((selected & neg).sum()/max(1,selected.sum()))
    logger.log({'phase':'edge_diag','edge_label_horizon':horizon,'num_edges':graph.num_edges,'num_pos_proxy':int(pos.sum()),'num_neg_proxy':int(neg.sum()),'reach_auc_proxy':auc,'reach_auprc_proxy':auprc,'selected_threshold':float(dcfg.get('edge_selected_threshold',0.5)),'selected_edges':int(selected.sum()),'false_positive_proxy_rate':fp,'reachable_edge_coverage_proxy':float((selected & pos).sum()/max(1,pos.sum()))})
def run_boundary_diagnostics(graph: BARSGraph, boundary: Optional[BoundaryIndex], cfg: Dict, logger: CSVLogger) -> None:
    if boundary is None: logger.log({'phase':'boundary_diag','enabled':0}); return
    psis=[]; supported=[]; out=graph.outgoing_edges()
    for eid in range(graph.num_edges):
        for ne in out[int(graph.dst[eid])]: ne=int(ne); psis.append(boundary.psi(eid,ne)); supported.append(int(boundary.has_arr[eid] and boundary.has_dep[ne]))
    if not psis: logger.log({'phase':'boundary_diag','enabled':1,'num_pairs':0}); return
    p=np.asarray(psis,dtype=np.float32); logger.log({'phase':'boundary_diag','enabled':1,'num_pairs':len(p),'psi_mean':float(p.mean()),'psi_p10':float(np.quantile(p,0.10)),'psi_p50':float(np.quantile(p,0.50)),'psi_p90':float(np.quantile(p,0.90)),'supported_pair_rate':float(np.mean(supported))})
def run_path_diagnostics(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, boundary: Optional[BoundaryIndex], cfg: Dict, logger: CSVLogger) -> None:
    dcfg=cfg.get('diagnostics',{}); variants=dcfg.get('planner_variants',['shortest','reachability','full_bars']); num_pairs=int(dcfg.get('num_path_pairs',256)); min_dt=int(dcfg.get('path_min_dt',cfg.get('reachability',{}).get('horizon',30)*2)); max_dt=int(dcfg.get('path_max_dt',250)); rng=np.random.default_rng(int(cfg.get('seed',0))+131); nbrs=NearestNeighbors(n_neighbors=1).fit(graph.node_embeddings); lambda_r=float(cfg.get('planner',{}).get('lambda_risk',1.0)); lambda_b=float(cfg.get('planner',{}).get('lambda_boundary',1.0))
    for pair_id in range(num_pairs):
        try: i,j,dt=dataset.sample_future_pairs(1,max_dt,rng,min_dt=min_dt)
        except Exception: break
        i=int(i[0]); j=int(j[0]); dt=int(dt[0]); s_node=int(nbrs.kneighbors(embeddings[i:i+1],return_distance=False)[0,0]); g_node=int(nbrs.kneighbors(embeddings[j:j+1],return_distance=False)[0,0])
        for variant in variants:
            result=plan_path(graph,s_node,g_node,variant=variant,lambda_risk=lambda_r,lambda_boundary=lambda_b,boundary=boundary)
            logger.log({'phase':'path_diag','pair_id':pair_id,'start_index':i,'goal_index':j,'true_future_dt':dt,'start_node':s_node,'goal_node':g_node,**result.to_row()})

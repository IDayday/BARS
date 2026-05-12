from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import numpy as np, torch
from sklearn.neighbors import NearestNeighbors
from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from bars.models.reachability import ReachabilityModel
from .types import BARSGraph, EDGE_KIND_KNN, EDGE_KIND_TEMPORAL

def _add_edge(edge_map: Dict[Tuple[int,int], int], s: int, d: int, kind: int) -> None:
    if s==d: return
    key=(int(s),int(d)); edge_map[key]=max(edge_map.get(key,kind), kind)
@torch.no_grad()
def _score_reachability(reach_model: Optional[ReachabilityModel], node_embeddings: np.ndarray, src: np.ndarray, dst: np.ndarray, device, batch_size: int = 32768, fallback_scale: Optional[float] = None) -> np.ndarray:
    if len(src)==0: return np.empty(0,dtype=np.float32)
    if reach_model is None:
        dist=np.linalg.norm(node_embeddings[src]-node_embeddings[dst],axis=1); scale=fallback_scale or (np.median(dist)+1e-6); return np.exp(-dist/scale).astype(np.float32)
    reach_model.eval(); probs=[]
    for st in range(0,len(src),batch_size):
        sl=slice(st,st+batch_size); zu=torch.as_tensor(node_embeddings[src[sl]],dtype=torch.float32,device=device); zv=torch.as_tensor(node_embeddings[dst[sl]],dtype=torch.float32,device=device); probs.append(reach_model.prob(zu,zv).float().cpu().numpy())
    return np.concatenate(probs,0).astype(np.float32)
def build_edges(dataset: OfflineDataset, embeddings: np.ndarray, node_indices: np.ndarray, reach_model: Optional[ReachabilityModel], cfg: Dict, device, logger: CSVLogger) -> BARSGraph:
    gcfg=cfg.get('graph',{}); node_emb=embeddings[node_indices].astype(np.float32); n=len(node_indices); edge_map={}
    logger.log({'phase':'edges','event':'start','num_nodes':n,'node_method':cfg.get('graph',{}).get('node_method','bars')})
    knn=int(gcfg.get('edge_knn',16))
    if n>1 and knn>0:
        ind=NearestNeighbors(n_neighbors=min(knn+1,n)).fit(node_emb).kneighbors(node_emb,return_distance=False)
        for i in range(n):
            for j in ind[i,1:]:
                _add_edge(edge_map,i,int(j),EDGE_KIND_KNN)
                if bool(gcfg.get('bidirectional_knn',True)): _add_edge(edge_map,int(j),i,EDGE_KIND_KNN)
        logger.log({'phase':'edges','event':'knn_candidates_built','candidate_edges':len(edge_map),'knn':knn})
    temporal_connect=int(gcfg.get('temporal_connect',4)); temporal_max_dt=int(gcfg.get('temporal_edge_horizon',80))
    if temporal_connect>0:
        by_traj=defaultdict(list)
        for ni,gi in enumerate(node_indices): by_traj[int(dataset.traj_id[gi])].append((int(dataset.timestep[gi]),ni,int(gi)))
        for arr in by_traj.values():
            arr.sort()
            for p,(_,ni,gi) in enumerate(arr):
                for q in range(p+1,min(len(arr),p+1+temporal_connect)):
                    _,nj,gj=arr[q]; dt=int(dataset.timestep[gj]-dataset.timestep[gi])
                    if 0<dt<=temporal_max_dt:
                        _add_edge(edge_map,ni,nj,EDGE_KIND_TEMPORAL)
                        if bool(gcfg.get('temporal_reverse',False)): _add_edge(edge_map,nj,ni,EDGE_KIND_TEMPORAL)
        logger.log({'phase':'edges','event':'temporal_candidates_built','candidate_edges':len(edge_map),'temporal_connect':temporal_connect,'temporal_edge_horizon':temporal_max_dt})
    if not edge_map: raise RuntimeError('No candidate edges constructed.')
    pairs=np.asarray(list(edge_map.keys()),dtype=np.int64); kind=np.asarray([edge_map[tuple(p)] for p in pairs],dtype=np.int32); src=pairs[:,0]; dst=pairs[:,1]
    dist=np.linalg.norm(node_emb[src]-node_emb[dst],axis=1).astype(np.float32); dist_scale=float(np.median(dist)+1e-6); cost=dist/dist_scale
    if str(gcfg.get('cost_mode','hybrid'))=='hybrid':
        same=dataset.traj_id[node_indices[src]]==dataset.traj_id[node_indices[dst]]; dt=dataset.timestep[node_indices[dst]]-dataset.timestep[node_indices[src]]; temporal=same & (dt>0) & (dt<=temporal_max_dt); cost[temporal]=np.log1p(dt[temporal]).astype(np.float32)/np.log1p(max(temporal_max_dt,1))
    p_exec=_score_reachability(reach_model,node_emb,src,dst,device,batch_size=int(gcfg.get('score_batch_size',32768)),fallback_scale=dist_scale); risk=(-np.log(np.clip(p_exec,float(gcfg.get('p_clip',1e-4)),1.0))).astype(np.float32)
    logger.log({'phase':'edges','event':'scored','candidate_edges':len(src),'p_exec_mean':float(np.mean(p_exec)) if len(p_exec) else 0.0,'risk_mean':float(np.mean(risk)) if len(risk) else 0.0})
    top_k=int(gcfg.get('top_outgoing',16)); prune_lambda=float(gcfg.get('prune_lambda_risk',0.25))
    if top_k>0:
        score=cost+prune_lambda*risk; keep=[]
        for s in range(n):
            ids=np.where(src==s)[0]
            if len(ids): keep.append(ids[np.argsort(score[ids])[:top_k]])
        keep_idx=np.concatenate(keep) if keep else np.empty(0,dtype=np.int64); src,dst,cost,risk,p_exec,kind=src[keep_idx],dst[keep_idx],cost[keep_idx],risk[keep_idx],p_exec[keep_idx],kind[keep_idx]
        logger.log({'phase':'edges','event':'pruned','top_outgoing':top_k,'prune_lambda_risk':prune_lambda,'kept_edges':len(src)})
    graph=BARSGraph(node_indices,node_emb,src,dst,cost,risk,p_exec,kind); logger.log({'phase':'edges','event':'completed','num_nodes':graph.num_nodes,'num_edges':graph.num_edges,'mean_out_degree':graph.num_edges/max(1,graph.num_nodes),'p_exec_mean':float(np.mean(p_exec)) if len(p_exec) else 0.0,'p_exec_p10':float(np.quantile(p_exec,0.10)) if len(p_exec) else 0.0,'risk_mean':float(np.mean(risk)) if len(risk) else 0.0,'cost_mean':float(np.mean(cost)) if len(cost) else 0.0}); return graph

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset

def farthest_point_sampling(x: np.ndarray, k: int, rng: np.random.Generator, start_idx: int | None = None) -> np.ndarray:
    n=len(x); k=min(int(k),n)
    if k<=0: return np.empty(0,dtype=np.int64)
    selected=np.empty(k,dtype=np.int64); selected[0]=int(start_idx if start_idx is not None else rng.integers(0,n)); min_dist=np.sum((x-x[selected[0]])**2,axis=1)
    for i in range(1,k): selected[i]=int(np.argmax(min_dist)); min_dist=np.minimum(min_dist, np.sum((x-x[selected[i]])**2,axis=1))
    return selected

def _nearest_to_centers(x, centers): return np.unique(NearestNeighbors(n_neighbors=1).fit(x).kneighbors(centers, return_distance=False)[:,0].astype(np.int64))
def _rank01(score):
    if len(score)==0: return score
    order=np.argsort(score); ranks=np.empty_like(order,dtype=np.float32); ranks[order]=np.linspace(0,1,len(score),dtype=np.float32); return ranks

def _build_knn_sparse(x: np.ndarray, k: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
    k_eff=min(k+1,len(x)); nbrs=NearestNeighbors(n_neighbors=k_eff).fit(x); dist,ind=nbrs.kneighbors(x,return_distance=True); rows=[]; cols=[]; vals=[]; sigma=np.median(dist[:,1:])+1e-6 if k_eff>1 else 1.0
    for i in range(len(x)):
        for d,j in zip(dist[i,1:], ind[i,1:]):
            w=float(np.exp(-(d**2)/(sigma**2))); rows.extend([i,int(j)]); cols.extend([int(j),i]); vals.extend([w,w])
    mat=sparse.coo_matrix((vals,(rows,cols)),shape=(len(x),len(x))).tocsr(); mat.eliminate_zeros(); return mat, ind[:,1:]

def spectral_bottleneck_scores(dataset: OfflineDataset, embeddings: np.ndarray, pool_idx: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict[str,float]]:
    gcfg=cfg.get('graph',{}); x=embeddings[pool_idx]; knn=int(gcfg.get('support_knn',12)); clusters=max(2,min(int(gcfg.get('spectral_clusters',8)),len(x)-1)); adj,neigh=_build_knn_sparse(x,knn); lap=csgraph.laplacian(adj,normed=True)
    try:
        vals,vecs=eigsh(lap,k=min(clusters,len(x)-2),which='SM'); feat=vecs[:,1:] if vecs.shape[1]>1 else vecs; labels=MiniBatchKMeans(n_clusters=clusters,random_state=int(cfg.get('seed',0)),batch_size=4096).fit_predict(feat)
    except Exception: labels=MiniBatchKMeans(n_clusters=clusters,random_state=int(cfg.get('seed',0)),batch_size=4096).fit_predict(x)
    boundary=np.zeros(len(x),dtype=np.float32)
    for i in range(len(x)): boundary[i]=float(np.mean(labels[neigh[i]]!=labels[i])) if len(neigh[i]) else 0.0
    pos={int(g):p for p,g in enumerate(pool_idx)}; transition=np.zeros(len(x),dtype=np.float32)
    for p,g in enumerate(pool_idx):
        q=pos.get(int(g)+1)
        if q is not None and dataset.traj_id[g]==dataset.traj_id[int(g)+1] and labels[p]!=labels[q]: transition[p]+=1; transition[q]+=1
    degree=np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32); score=_rank01(boundary)+_rank01(transition)+0.25*_rank01(1/(degree+1e-6))
    return score.astype(np.float32), {'support_nodes':float(len(x)),'support_edges':float(adj.nnz//2),'boundary_mean':float(boundary.mean()),'transition_cross_mean':float(transition.mean())}

def select_graph_nodes(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, logger: CSVLogger) -> np.ndarray:
    gcfg=cfg.get('graph',{}); method=str(gcfg.get('node_method','bars')).lower(); num_nodes=int(gcfg.get('num_nodes',500)); rng=np.random.default_rng(int(cfg.get('seed',0))+101); max_support=min(int(gcfg.get('max_support_states',20000)),dataset.size); pool_idx=rng.choice(dataset.size,size=max_support,replace=False) if max_support<dataset.size else np.arange(dataset.size); pool_emb=embeddings[pool_idx]
    if method=='random': chosen=rng.choice(dataset.size,size=min(num_nodes,dataset.size),replace=False); logger.log({'phase':'nodes','node_method':method,'num_nodes':len(chosen)}); return chosen.astype(np.int64)
    if method=='fps': chosen=pool_idx[farthest_point_sampling(pool_emb,num_nodes,rng)]; logger.log({'phase':'nodes','node_method':method,'num_nodes':len(chosen)}); return chosen.astype(np.int64)
    if method=='kmeans':
        km=MiniBatchKMeans(n_clusters=min(num_nodes,len(pool_idx)),random_state=int(cfg.get('seed',0)),batch_size=4096,n_init=3).fit(pool_emb); chosen=pool_idx[_nearest_to_centers(pool_emb,km.cluster_centers_)]; logger.log({'phase':'nodes','node_method':method,'num_nodes':len(chosen)}); return chosen.astype(np.int64)
    if method in {'spectral','bars'}:
        score,metrics=spectral_bottleneck_scores(dataset,embeddings,pool_idx,cfg); nb=min(int(gcfg.get('num_bottleneck_nodes',max(1,int(0.35*num_nodes)))),num_nodes,len(pool_idx)); top_local=np.argsort(-score)[:nb]; bottleneck=pool_idx[top_local]
        if method=='spectral': chosen=bottleneck
        else:
            remaining=max(0,num_nodes-len(bottleneck)); mask=np.ones(len(pool_idx),dtype=bool); mask[top_local]=False; cand_idx=pool_idx[mask]; cand_emb=embeddings[cand_idx]
            if remaining>0 and len(cand_idx)>0:
                dist_to_b=np.min(((cand_emb[:,None,:]-embeddings[bottleneck][None,:,:])**2).sum(-1),axis=1); keep=dist_to_b>np.quantile(dist_to_b,float(gcfg.get('anchor_exclude_quantile',0.10)))
                if keep.sum()>=remaining: cand_idx=cand_idx[keep]; cand_emb=cand_emb[keep]
                anchors=cand_idx[farthest_point_sampling(cand_emb,remaining,rng)]; chosen=np.unique(np.concatenate([bottleneck,anchors]))
            else: chosen=bottleneck
        logger.log({'phase':'nodes','node_method':method,'num_nodes':len(chosen),**metrics}); return chosen[:num_nodes].astype(np.int64)
    raise ValueError(f'Unknown graph.node_method={method}')

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple
import numpy as np, torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import trange
from bars.common.checkpoint import load_checkpoint, save_checkpoint
from bars.common.logging import CSVLogger
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.models.reachability import ReachabilityModel

def _build_latent_hard_negative_pool(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, rng: np.random.Generator) -> Optional[Tuple[np.ndarray,np.ndarray]]:
    rcfg=cfg.get('reachability',{})
    if not bool(rcfg.get('use_hard_neg',True)): return None
    n_pool=min(int(rcfg.get('hard_neg_pool',20000)), dataset.size); k=int(rcfg.get('hard_neg_knn',16)); idx=rng.choice(dataset.size,size=n_pool,replace=False)
    neigh=NearestNeighbors(n_neighbors=min(k+1,n_pool)).fit(embeddings[idx]).kneighbors(embeddings[idx], return_distance=False); src=[]; dst=[]
    for row, ns in enumerate(neigh):
        i=int(idx[row])
        for n in ns[1:]:
            j=int(idx[n])
            if dataset.traj_id[i] != dataset.traj_id[j]: src.append(i); dst.append(j); break
    if not src: return None
    return np.asarray(src,dtype=np.int64), np.asarray(dst,dtype=np.int64)

def train_reachability(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, run_dir: str, device, logger: CSVLogger, stopper: Optional[Stopper] = None) -> ReachabilityModel:
    rcfg=cfg.get('reachability',{}); ckpt_path=os.path.join(run_dir,'checkpoints','reachability.pt')
    model=ReachabilityModel(embeddings.shape[1], tuple(rcfg.get('hidden_dims',[256,256]))).to(device)
    if rcfg.get('load_if_exists', True) and os.path.exists(ckpt_path): load_checkpoint(ckpt_path, model, map_location=str(device)); logger.log({'phase':'reachability','event':'loaded','path':ckpt_path}); return model
    opt=torch.optim.AdamW(model.parameters(), lr=float(rcfg.get('lr',3e-4)), weight_decay=float(rcfg.get('weight_decay',1e-5)))
    batch_size=int(rcfg.get('batch_size',2048)); steps=int(rcfg.get('steps',20000)); horizon=int(rcfg.get('horizon',30)); log_every=int(rcfg.get('log_every',500)); hard_weight=float(rcfg.get('hard_neg_weight',0.5)); rng=np.random.default_rng(int(cfg.get('seed',0))+43)
    hard_pool=_build_latent_hard_negative_pool(dataset,embeddings,cfg,rng); z_all=torch.as_tensor(embeddings,dtype=torch.float32,device=device)
    for step in trange(steps, desc='train_reachability', dynamic_ncols=True):
        if stopper is not None and stopper.stop_requested: break
        half=batch_size//2; ip,jp,_=dataset.sample_future_pairs(half,horizon,rng); ineg=dataset.sample_indices(half,rng); jneg=dataset.sample_indices(half,rng)
        same=dataset.traj_id[ineg]==dataset.traj_id[jneg]; tries=0
        while same.any() and tries<5: jneg[same]=dataset.sample_indices(int(same.sum()),rng); same=dataset.traj_id[ineg]==dataset.traj_id[jneg]; tries+=1
        src=np.concatenate([ip,ineg]); dst=np.concatenate([jp,jneg]); labels_np=np.concatenate([np.ones(half,dtype=np.float32),np.zeros(half,dtype=np.float32)]); weights_np=np.ones(len(labels_np),dtype=np.float32)
        if hard_pool is not None and hard_weight>0:
            hs,hd=hard_pool; hn=min(half,len(hs)); hi=rng.choice(len(hs),size=hn,replace=len(hs)<hn); src=np.concatenate([src,hs[hi]]); dst=np.concatenate([dst,hd[hi]]); labels_np=np.concatenate([labels_np,np.zeros(hn,dtype=np.float32)]); weights_np=np.concatenate([weights_np,np.full(hn,hard_weight,dtype=np.float32)])
        src_t=torch.as_tensor(src,dtype=torch.long,device=device); dst_t=torch.as_tensor(dst,dtype=torch.long,device=device); labels=torch.as_tensor(labels_np,dtype=torch.float32,device=device); weights=torch.as_tensor(weights_np,dtype=torch.float32,device=device)
        opt.zero_grad(set_to_none=True); logits=model(z_all[src_t], z_all[dst_t]); loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(rcfg.get('grad_clip',10.0))); opt.step()
        if step % log_every == 0 or step==steps-1:
            with torch.no_grad(): prob=torch.sigmoid(logits); pos_prob=prob[labels>0.5].mean().item(); neg_prob=prob[labels<0.5].mean().item()
            logger.log({'phase':'reachability','step':step,'loss':float(loss.detach().cpu()),'pos_prob':pos_prob,'neg_prob':neg_prob,'hard_pool_size':0 if hard_pool is None else len(hard_pool[0])})
    save_checkpoint(ckpt_path, model, optimizer=opt, latent_dim=embeddings.shape[1]); logger.log({'phase':'reachability','event':'saved','path':ckpt_path}); return model

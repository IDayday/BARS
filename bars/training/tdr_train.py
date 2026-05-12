from __future__ import annotations
import os
from typing import Dict, Optional
import numpy as np, torch
import torch.nn.functional as F
from tqdm import trange
from bars.common.checkpoint import load_checkpoint, save_checkpoint
from bars.common.logging import CSVLogger
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.models.tdr import TemporalDistanceModel

def train_tdr(dataset: OfflineDataset, cfg: Dict, run_dir: str, device, logger: CSVLogger, stopper: Optional[Stopper] = None) -> TemporalDistanceModel:
    tcfg=cfg.get('tdr',{}); ckpt_path=os.path.join(run_dir,'checkpoints','tdr.pt')
    model=TemporalDistanceModel(dataset.obs_dim, int(tcfg.get('latent_dim',32)), tuple(tcfg.get('hidden_dims',[256,256])), tuple(tcfg.get('pair_hidden_dims',[256,256]))).to(device)
    if tcfg.get('load_if_exists', True) and os.path.exists(ckpt_path): load_checkpoint(ckpt_path, model, map_location=str(device)); logger.log({'phase':'tdr','event':'loaded','path':ckpt_path}); return model
    opt=torch.optim.AdamW(model.parameters(), lr=float(tcfg.get('lr',3e-4)), weight_decay=float(tcfg.get('weight_decay',1e-4)))
    batch_size=int(tcfg.get('batch_size',1024)); steps=int(tcfg.get('steps',20000)); horizon=int(tcfg.get('horizon',50)); log_every=int(tcfg.get('log_every',500)); rng=np.random.default_rng(int(cfg.get('seed',0))+17)
    scaler=torch.cuda.amp.GradScaler(enabled=bool(tcfg.get('amp',True)) and device.type=='cuda'); obs_norm=dataset.obs_normalizer.encode(dataset.observations)
    for step in trange(steps, desc='train_tdr', dynamic_ncols=True):
        if stopper is not None and stopper.stop_requested: break
        i,j,dt=dataset.sample_future_pairs(batch_size,horizon,rng); ni=dataset.sample_indices(batch_size,rng); nj=dataset.sample_indices(batch_size,rng)
        obs_i=torch.as_tensor(obs_norm[i],dtype=torch.float32,device=device); obs_j=torch.as_tensor(obs_norm[j],dtype=torch.float32,device=device); obs_ni=torch.as_tensor(obs_norm[ni],dtype=torch.float32,device=device); obs_nj=torch.as_tensor(obs_norm[nj],dtype=torch.float32,device=device); target=torch.as_tensor(np.log1p(dt).astype(np.float32),dtype=torch.float32,device=device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            _,_,pred,pos_logit=model(obs_i,obs_j); _,_,_,neg_logit=model(obs_ni,obs_nj)
            dist_loss=F.smooth_l1_loss(pred,target); logits=torch.cat([pos_logit,neg_logit]); labels=torch.cat([torch.ones_like(pos_logit),torch.zeros_like(neg_logit)]); bce=F.binary_cross_entropy_with_logits(logits,labels); loss=dist_loss+float(tcfg.get('bce_weight',1.0))*bce
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get('grad_clip',10.0))); scaler.step(opt); scaler.update()
        if step % log_every == 0 or step == steps-1: logger.log({'phase':'tdr','step':step,'loss':float(loss.detach().cpu()),'dist_loss':float(dist_loss.detach().cpu()),'bce_loss':float(bce.detach().cpu())})
    save_checkpoint(ckpt_path, model, optimizer=opt, obs_mean=dataset.obs_normalizer.mean, obs_std=dataset.obs_normalizer.std); logger.log({'phase':'tdr','event':'saved','path':ckpt_path}); return model

@torch.no_grad()
def embed_dataset(model: TemporalDistanceModel, dataset: OfflineDataset, device, batch_size: int = 8192, cache_path: Optional[str] = None, force: bool = False) -> np.ndarray:
    if cache_path and os.path.exists(cache_path) and not force: return np.load(cache_path)
    model.eval(); obs_norm=dataset.obs_normalizer.encode(dataset.observations); out=[]
    for st in range(0, dataset.size, batch_size):
        x=torch.as_tensor(obs_norm[st:st+batch_size],dtype=torch.float32,device=device); out.append(model.encode(x).float().cpu().numpy())
    emb=np.concatenate(out,0).astype(np.float32)
    if cache_path: os.makedirs(os.path.dirname(cache_path),exist_ok=True); np.save(cache_path, emb)
    return emb

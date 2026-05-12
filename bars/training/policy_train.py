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
from bars.models.policy import GoalConditionedPolicy

def train_policy(dataset: OfflineDataset, cfg: Dict, run_dir: str, device, logger: CSVLogger, stopper: Optional[Stopper] = None) -> GoalConditionedPolicy:
    pcfg=cfg.get('policy',{}); ckpt_path=os.path.join(run_dir,'checkpoints','policy.pt')
    model=GoalConditionedPolicy(dataset.obs_dim,dataset.action_dim,tuple(pcfg.get('hidden_dims',[256,256,256])),bool(pcfg.get('goal_delta',True))).to(device)
    if pcfg.get('load_if_exists', True) and os.path.exists(ckpt_path): load_checkpoint(ckpt_path, model, map_location=str(device)); logger.log({'phase':'policy','event':'loaded','path':ckpt_path}); return model
    opt=torch.optim.AdamW(model.parameters(), lr=float(pcfg.get('lr',3e-4)), weight_decay=float(pcfg.get('weight_decay',1e-5)))
    batch_size=int(pcfg.get('batch_size',1024)); steps=int(pcfg.get('steps',30000)); horizon=int(pcfg.get('horizon',30)); log_every=int(pcfg.get('log_every',500)); rng=np.random.default_rng(int(cfg.get('seed',0))+29)
    scaler=torch.cuda.amp.GradScaler(enabled=bool(pcfg.get('amp',True)) and device.type=='cuda'); obs_norm=dataset.obs_normalizer.encode(dataset.observations); actions=dataset.actions.astype(np.float32)
    for step in trange(steps, desc='train_policy', dynamic_ncols=True):
        if stopper is not None and stopper.stop_requested: break
        i,j,_=dataset.sample_future_pairs(batch_size,horizon,rng); obs=torch.as_tensor(obs_norm[i],dtype=torch.float32,device=device); goal=torch.as_tensor(obs_norm[j],dtype=torch.float32,device=device); act=torch.as_tensor(actions[i],dtype=torch.float32,device=device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()): loss=F.mse_loss(model(obs,goal), act)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), float(pcfg.get('grad_clip',10.0))); scaler.step(opt); scaler.update()
        if step % log_every == 0 or step == steps-1: logger.log({'phase':'policy','step':step,'loss':float(loss.detach().cpu())})
    save_checkpoint(ckpt_path, model, optimizer=opt, obs_mean=dataset.obs_normalizer.mean, obs_std=dataset.obs_normalizer.std); logger.log({'phase':'policy','event':'saved','path':ckpt_path}); return model

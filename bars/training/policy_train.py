from __future__ import annotations
import os
from typing import Dict, Optional
import numpy as np, torch
import torch.nn.functional as F
from tqdm import trange
from bars.common.checkpoint import load_checkpoint, save_checkpoint
from bars.common.logging import CSVLogger
from bars.common.progress import tqdm_kwargs
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.models.external_policy import build_external_policy_from_config
from bars.models.policy import GoalConditionedPolicy
from bars.models.bars_iql import BARSIQLPolicy
from bars.training.bars_iql_train import train_bars_iql_policy

def train_policy(dataset: OfflineDataset, cfg: Dict, run_dir: str, device, logger: CSVLogger, stopper: Optional[Stopper] = None, embeddings=None):
    pcfg=cfg.get('policy',{})
    policy_type = str(pcfg.get('type', 'gcbc')).lower()
    if policy_type in {'bars_iql', 'bars-low', 'bars_low', 'iql', 'bars_iql_low'}:
        return train_bars_iql_policy(dataset, cfg, run_dir, device, logger, stopper=stopper, embeddings=embeddings)
    if policy_type in {'external', 'hiql_external', 'gas_external'}:
        model = build_external_policy_from_config(cfg, dataset, device=device)
        logger.log({
            'phase':'policy',
            'event':'external_loaded',
            'policy_type':pcfg.get('type'),
            'external_policy': str(cfg.get('external_policy', {}).get('factory', cfg.get('external_policy', {}).get('object', ''))),
            'checkpoint_path': str(cfg.get('external_policy', {}).get('checkpoint_path', '')),
        })
        return model
    ckpt_path=os.path.join(run_dir,'checkpoints','policy.pt')
    model=GoalConditionedPolicy(dataset.obs_dim,dataset.action_dim,tuple(pcfg.get('hidden_dims',[256,256,256])),bool(pcfg.get('goal_delta',True))).to(device)
    if pcfg.get('load_if_exists', True) and os.path.exists(ckpt_path): load_checkpoint(ckpt_path, model, map_location=str(device)); logger.log({'phase':'policy','event':'loaded','path':ckpt_path}); return model
    opt=torch.optim.AdamW(model.parameters(), lr=float(pcfg.get('lr',3e-4)), weight_decay=float(pcfg.get('weight_decay',1e-5)))
    batch_size=int(pcfg.get('batch_size',1024)); steps=int(pcfg.get('steps',30000)); horizon=int(pcfg.get('horizon',30)); log_every=int(pcfg.get('log_every',500)); rng=np.random.default_rng(int(cfg.get('seed',0))+29)
    scaler=torch.cuda.amp.GradScaler(enabled=bool(pcfg.get('amp',True)) and device.type=='cuda'); obs_norm=dataset.obs_normalizer.encode(dataset.observations); actions=dataset.actions.astype(np.float32, copy=False)
    cache_default = bool(getattr(device, 'type', str(device)) == 'cuda')
    cache_on_device = bool(pcfg.get('cache_dataset_on_device', cache_default))
    cache_max_mb = float(pcfg.get('cache_dataset_max_mb', 512))
    cache_mb = float((obs_norm.nbytes + actions.nbytes) / (1024 * 1024))
    obs_all = act_all = None
    if cache_on_device and cache_mb <= cache_max_mb:
        obs_all = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
        act_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
        logger.log({'phase':'policy','event':'dataset_cached_on_device','cache_dataset_mb':cache_mb,'cache_max_mb':cache_max_mb})
    for step in trange(steps, desc='train_policy', **tqdm_kwargs(cfg)):
        if stopper is not None and stopper.stop_requested: break
        i,j,_=dataset.sample_future_pairs(batch_size,horizon,rng)
        if obs_all is not None:
            i_t=torch.as_tensor(i,dtype=torch.long,device=device); j_t=torch.as_tensor(j,dtype=torch.long,device=device); obs=obs_all[i_t]; goal=obs_all[j_t]; act=act_all[i_t]
        else:
            obs=torch.as_tensor(obs_norm[i],dtype=torch.float32,device=device); goal=torch.as_tensor(obs_norm[j],dtype=torch.float32,device=device); act=torch.as_tensor(actions[i],dtype=torch.float32,device=device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()): loss=F.mse_loss(model(obs,goal), act)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), float(pcfg.get('grad_clip',10.0))); scaler.step(opt); scaler.update()
        if step % log_every == 0 or step == steps-1: logger.log({'phase':'policy','step':step,'loss':float(loss.detach().cpu())})
    save_checkpoint(ckpt_path, model, optimizer=opt, obs_mean=dataset.obs_normalizer.mean, obs_std=dataset.obs_normalizer.std); logger.log({'phase':'policy','event':'saved','path':ckpt_path}); return model

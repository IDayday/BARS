from __future__ import annotations
from typing import Dict, Optional
import numpy as np, torch
from sklearn.neighbors import NearestNeighbors
from bars.common.logging import CSVLogger
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.graph.boundary import BoundaryIndex
from bars.graph.planner import plan_path
from bars.graph.types import BARSGraph
from bars.models.policy import GoalConditionedPolicy
from bars.models.tdr import TemporalDistanceModel

def _reset_env(env):
    out=env.reset(); return out[0] if isinstance(out,tuple) else out
def _step_env(env,action):
    out=env.step(action)
    if len(out)==5:
        obs,rew,terminated,truncated,info=out; return obs,rew,bool(terminated or truncated),info
    return out
def _goal_obs_from_env(env, obs: np.ndarray, dataset: OfflineDataset, goal_dim: int, rng: np.random.Generator) -> np.ndarray:
    goal_obs=np.array(obs,copy=True); target=None
    for attr in ['target_goal','goal','_target_goal']:
        if hasattr(env,attr):
            try: target=np.asarray(getattr(env,attr),dtype=np.float32).reshape(-1); break
            except Exception: pass
    if target is not None and len(target)>=goal_dim: goal_obs[:goal_dim]=target[:goal_dim]
    else: goal_obs=np.array(dataset.observations[int(rng.integers(0,dataset.size))],copy=True)
    return goal_obs.astype(np.float32)
@torch.no_grad()
def _embed_one(model: TemporalDistanceModel, obs: np.ndarray, dataset: OfflineDataset, device) -> np.ndarray:
    x=torch.as_tensor(dataset.obs_normalizer.encode(obs[None]),dtype=torch.float32,device=device); return model.encode(x).float().cpu().numpy()[0]
def evaluate_planner_policy(env, dataset: OfflineDataset, tdr_model: TemporalDistanceModel, policy: GoalConditionedPolicy, graph: BARSGraph, boundary: Optional[BoundaryIndex], cfg: Dict, device, logger: CSVLogger, stopper: Optional[Stopper] = None) -> None:
    ecfg=cfg.get('eval',{})
    if not bool(ecfg.get('enabled',False)): logger.log({'phase':'eval','enabled':0}); return
    episodes=int(ecfg.get('episodes',20)); max_steps=int(ecfg.get('max_steps',1000)); subgoal_horizon=int(ecfg.get('subgoal_horizon',cfg.get('policy',{}).get('horizon',30))); success_threshold=float(ecfg.get('success_threshold',0.5)); subgoal_threshold=float(ecfg.get('subgoal_threshold',success_threshold)); goal_dim=int(ecfg.get('goal_dim',2)); variant=str(ecfg.get('variant',cfg.get('planner',{}).get('variant','full_bars'))); lambda_r=float(cfg.get('planner',{}).get('lambda_risk',1.0)); lambda_b=float(cfg.get('planner',{}).get('lambda_boundary',1.0)); rng=np.random.default_rng(int(cfg.get('seed',0))+211); nbrs=NearestNeighbors(n_neighbors=1).fit(graph.node_embeddings); action_low=getattr(env.action_space,'low',None); action_high=getattr(env.action_space,'high',None)
    for ep in range(episodes):
        if stopper is not None and stopper.stop_requested: break
        obs=np.asarray(_reset_env(env),dtype=np.float32); goal_obs=_goal_obs_from_env(env,obs,dataset,goal_dim,rng); total_reward=0.0; success=False; no_path_count=0; replans=0; steps=0; last_plan_edges=0
        while steps<max_steps:
            if np.linalg.norm(obs[:goal_dim]-goal_obs[:goal_dim])<=success_threshold: success=True; break
            z_s=_embed_one(tdr_model,obs,dataset,device); z_g=_embed_one(tdr_model,goal_obs,dataset,device); s_node=int(nbrs.kneighbors(z_s[None],return_distance=False)[0,0]); g_node=int(nbrs.kneighbors(z_g[None],return_distance=False)[0,0]); plan=plan_path(graph,s_node,g_node,variant=variant,lambda_risk=lambda_r,lambda_boundary=lambda_b,boundary=boundary); replans+=1
            if not plan.found or len(plan.node_path)<=1: no_path_count+=1; break
            last_plan_edges=len(plan.edge_path); next_node=plan.node_path[1]; subgoal_obs=dataset.observations[graph.node_indices[next_node]]; subgoal_obs=goal_obs if next_node==g_node else subgoal_obs
            for _ in range(subgoal_horizon):
                if stopper is not None and stopper.stop_requested: break
                action=policy.act(obs,subgoal_obs,dataset.obs_normalizer,action_low,action_high,device=str(device)); obs,rew,done,info=_step_env(env,action); obs=np.asarray(obs,dtype=np.float32); total_reward+=float(rew); steps+=1
                info_success=bool(info.get('success',False) or info.get('goal_achieved',False) or info.get('is_success',False))
                if info_success or np.linalg.norm(obs[:goal_dim]-goal_obs[:goal_dim])<=success_threshold: success=True; break
                if np.linalg.norm(obs[:goal_dim]-subgoal_obs[:goal_dim])<=subgoal_threshold: break
                if done or steps>=max_steps: break
            if success or steps>=max_steps: break
        logger.log({'phase':'eval','enabled':1,'episode':ep,'variant':variant,'success':int(success),'return':total_reward,'steps':steps,'replans':replans,'no_path_count':no_path_count,'last_plan_edges':last_plan_edges,'goal_distance_final':float(np.linalg.norm(obs[:goal_dim]-goal_obs[:goal_dim]))})

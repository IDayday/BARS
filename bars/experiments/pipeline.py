from __future__ import annotations
import os, socket, time
from typing import Dict, Optional
from bars.common.artifacts import package_logs
from bars.common.config import save_json
from bars.common.device import describe_visible_cuda, get_torch_device
from bars.common.logging import CSVLogger
from bars.common.seed import set_seed
from bars.common.stopper import Stopper
from bars.data.d4rl_dataset import load_d4rl_dataset
from bars.data.toy_dataset import make_toy_dataset
from bars.eval.rollout import evaluate_planner_policy
from bars.graph.boundary import BoundaryIndex, build_boundary_index
from bars.graph.diagnostics import run_boundary_diagnostics, run_edge_diagnostics, run_path_diagnostics
from bars.graph.edges import build_edges
from bars.graph.nodes import select_graph_nodes
from bars.graph.types import BARSGraph
from bars.training.policy_train import train_policy
from bars.training.reach_train import train_reachability
from bars.training.tdr_train import embed_dataset, train_tdr

def _make_loggers(run_dir: str, cfg: Dict) -> Dict[str, CSVLogger]:
    base={'run_id':cfg.get('run_id',os.path.basename(run_dir)),'env':cfg.get('env_name',cfg.get('data',{}).get('env_name','unknown')),'seed':cfg.get('seed',0),'variant':cfg.get('planner',{}).get('variant','full_bars')}
    return {name: CSVLogger(os.path.join(run_dir,'logs',f'{name}.csv'), base) for name in ['train','graph','diagnostics','eval','summary']}
def _load_data(cfg: Dict):
    source=str(cfg.get('data',{}).get('source','d4rl')).lower()
    if source=='toy':
        dc=cfg.get('data',{})
        return None, make_toy_dataset(num_traj=int(dc.get('num_traj',16)), length=int(dc.get('length',20)), seed=int(cfg.get('seed',0)))
    if source=='d4rl': return load_d4rl_dataset(cfg.get('data',{}).get('env_name',cfg.get('env_name','antmaze-medium-play-v2')), dataset_limit=int(cfg.get('data',{}).get('dataset_limit',0)))
    raise ValueError(f'Unknown data.source={source}')
def run_experiment(cfg: Dict, run_dir: str, stopper: Optional[Stopper] = None) -> str:
    os.makedirs(run_dir,exist_ok=True); cfg=dict(cfg); cfg.setdefault('run_id',os.path.basename(run_dir)); cfg.setdefault('env_name',cfg.get('data',{}).get('env_name','unknown')); save_json(cfg, os.path.join(run_dir,'config.json')); loggers=_make_loggers(run_dir,cfg); summary=loggers['summary']; start=time.time(); status='started'
    try:
        set_seed(int(cfg.get('seed',0))); device=get_torch_device(str(cfg.get('device','cuda'))); summary.log({'phase':'run','status':'started','host':socket.gethostname(),'cuda':describe_visible_cuda(),'run_dir':run_dir})
        env,dataset=_load_data(cfg); summary.log({'phase':'data','status':'loaded','dataset_size':dataset.size,'num_trajectories':dataset.num_trajectories,'obs_dim':dataset.obs_dim,'action_dim':dataset.action_dim})
        if stopper is not None and stopper.stop_requested: status='stopped_before_training'; return run_dir
        tdr_model=train_tdr(dataset,cfg,run_dir,device,loggers['train'],stopper); emb_cache=os.path.join(run_dir,'cache','embeddings.npy'); embeddings=embed_dataset(tdr_model,dataset,device,batch_size=int(cfg.get('tdr',{}).get('embed_batch_size',8192)),cache_path=emb_cache,force=bool(cfg.get('tdr',{}).get('force_reembed',False))); summary.log({'phase':'tdr','status':'embedded','embedding_path':emb_cache,'latent_dim':embeddings.shape[1]})
        if stopper is not None and stopper.stop_requested: status='stopped_after_tdr'; return run_dir
        policy=train_policy(dataset,cfg,run_dir,device,loggers['train'],stopper)
        if stopper is not None and stopper.stop_requested: status='stopped_after_policy'; return run_dir
        reach_model=train_reachability(dataset,embeddings,cfg,run_dir,device,loggers['train'],stopper) if bool(cfg.get('reachability',{}).get('enabled',True)) else None
        if stopper is not None and stopper.stop_requested: status='stopped_after_reachability'; return run_dir
        graph_path=os.path.join(run_dir,'cache','graph.npz')
        if bool(cfg.get('graph',{}).get('load_if_exists',True)) and os.path.exists(graph_path): graph=BARSGraph.load_npz(graph_path); loggers['graph'].log({'phase':'graph','event':'loaded','path':graph_path,'num_nodes':graph.num_nodes,'num_edges':graph.num_edges})
        else:
            node_indices=select_graph_nodes(dataset,embeddings,cfg,loggers['graph']); graph=build_edges(dataset,embeddings,node_indices,reach_model,cfg,device,loggers['graph']); graph.save_npz(graph_path); loggers['graph'].log({'phase':'graph','event':'saved','path':graph_path,'num_nodes':graph.num_nodes,'num_edges':graph.num_edges})
        if stopper is not None and stopper.stop_requested: status='stopped_after_graph'; return run_dir
        boundary=None; boundary_path=os.path.join(run_dir,'cache','boundary.npz')
        if bool(cfg.get('boundary',{}).get('enabled',True)):
            if bool(cfg.get('boundary',{}).get('load_if_exists',True)) and os.path.exists(boundary_path): boundary=BoundaryIndex.load_npz(boundary_path); loggers['graph'].log({'phase':'boundary','event':'loaded','path':boundary_path})
            else: boundary=build_boundary_index(dataset,embeddings,graph,cfg,loggers['graph']); boundary.save_npz(boundary_path); loggers['graph'].log({'phase':'boundary','event':'saved','path':boundary_path})
        if bool(cfg.get('diagnostics',{}).get('enabled',True)): run_edge_diagnostics(dataset,graph,cfg,loggers['diagnostics']); run_boundary_diagnostics(graph,boundary,cfg,loggers['diagnostics']); run_path_diagnostics(dataset,embeddings,graph,boundary,cfg,loggers['diagnostics'])
        if stopper is not None and stopper.stop_requested: status='stopped_after_diagnostics'; return run_dir
        if env is not None: evaluate_planner_policy(env,dataset,tdr_model,policy,graph,boundary,cfg,device,loggers['eval'],stopper)
        status='completed' if not (stopper and stopper.stop_requested) else f'stopped_{stopper.reason}'; return run_dir
    except Exception as exc:
        status='failed'; summary.log({'phase':'run','status':status,'error':repr(exc)}); raise
    finally:
        elapsed=time.time()-start
        try:
            summary.log({'phase':'run','status':status,'elapsed_sec':elapsed}); save_json({'run_id':cfg.get('run_id',os.path.basename(run_dir)),'status':status,'elapsed_sec':elapsed,'run_dir':run_dir,'created_at':time.time()}, os.path.join(run_dir,'manifest.json')); archive=package_logs(run_dir); summary.log({'phase':'run','status':'packaged','archive_path':archive})
        except Exception: pass

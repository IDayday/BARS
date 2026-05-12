from __future__ import annotations
import argparse, os, time
from bars.common.artifacts import package_logs
from bars.common.config import apply_dotlist, load_json
from bars.common.stopper import GLOBAL_STOPPER, install_signal_handlers
from bars.experiments.pipeline import run_experiment

def _default_run_dir(root: str, cfg: dict) -> str:
    env=cfg.get('data',{}).get('env_name',cfg.get('env_name','unknown_env')); variant=cfg.get('planner',{}).get('variant','full_bars'); seed=cfg.get('seed',0); stamp=time.strftime('%Y%m%d_%H%M%S'); return os.path.join(root,str(env),str(variant),f'seed{seed}_{stamp}')
def main(argv=None) -> None:
    parser=argparse.ArgumentParser(prog='bars'); sub=parser.add_subparsers(dest='cmd',required=True)
    run_p=sub.add_parser('run'); run_p.add_argument('--config',required=True); run_p.add_argument('--run-dir',default=None); run_p.add_argument('--log-root',default='runs'); run_p.add_argument('--env',dest='env_name',default=None); run_p.add_argument('--seed',type=int,default=None); run_p.add_argument('--variant',default=None); run_p.add_argument('--node-method',default=None); run_p.add_argument('--set',action='append',default=[])
    pack_p=sub.add_parser('pack'); pack_p.add_argument('--run-dir',required=True)
    args=parser.parse_args(argv)
    if args.cmd=='pack': print(package_logs(args.run_dir)); return
    cfg=load_json(args.config)
    if args.env_name is not None: cfg.setdefault('data',{})['env_name']=args.env_name; cfg['env_name']=args.env_name
    if args.seed is not None: cfg['seed']=args.seed
    if args.variant is not None: cfg.setdefault('planner',{})['variant']=args.variant; cfg.setdefault('eval',{})['variant']=args.variant
    if args.node_method is not None: cfg.setdefault('graph',{})['node_method']=args.node_method
    cfg=apply_dotlist(cfg,args.set); run_dir=args.run_dir or _default_run_dir(args.log_root,cfg); cfg['run_id']=os.path.basename(run_dir); install_signal_handlers(GLOBAL_STOPPER); run_experiment(cfg,run_dir,stopper=GLOBAL_STOPPER)
if __name__=='__main__': main()

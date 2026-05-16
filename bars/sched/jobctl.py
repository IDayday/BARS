from __future__ import annotations
import argparse, itertools, json, os, signal, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Optional
from bars.common.artifacts import package_logs
from bars.common.config import load_json, save_json
from bars.common.logging import read_last_csv_row
from bars.sched.gpu import parse_gpu_list, query_gpus

def _jobs_dir(log_root: str) -> Path: p=Path(log_root)/'_jobs'; p.mkdir(parents=True,exist_ok=True); return p
def _state_path(log_root: str, run_id: str) -> Path: return _jobs_dir(log_root)/f'{run_id}.json'
def _scheduler_pid_path(log_root: str) -> Path: return _jobs_dir(log_root)/'scheduler.pid'
def _read_state(path: Path) -> Dict:
    with open(path,'r',encoding='utf-8') as f: return json.load(f)
def _write_state(log_root: str, state: Dict) -> None: save_json(state, str(_state_path(log_root,state['run_id'])))
def _pid_alive(pid: int) -> bool:
    if pid<=0: return False
    try: os.kill(pid,0); return True
    except OSError: return False
def _archive_exists(run_dir: str) -> bool:
    return bool(run_dir and list((Path(run_dir)/'archives').glob('*.tar.gz')))
def _last_log_update(run_dir: str) -> Dict[str, object]:
    if not run_dir: return {'last_log_update_time':None,'last_log_update_age_min':None,'last_log_update_path':'','stale':False}
    paths=[Path(run_dir)/'stdout.log',Path(run_dir)/'stderr.log',Path(run_dir)/'logs'/'summary.csv',Path(run_dir)/'logs'/'graph.csv',Path(run_dir)/'logs'/'diagnostics.csv',Path(run_dir)/'logs'/'eval.csv']
    existing=[p for p in paths if p.exists()]
    if not existing: return {'last_log_update_time':None,'last_log_update_age_min':None,'last_log_update_path':'','stale':False}
    latest=max(existing,key=lambda p:p.stat().st_mtime); age_min=max(0.0,(time.time()-latest.stat().st_mtime)/60.0)
    return {'last_log_update_time':latest.stat().st_mtime,'last_log_update_age_min':age_min,'last_log_update_path':str(latest),'stale':age_min>=90.0}
def _state_with_summary_fallback(st: Dict) -> Dict:
    out=dict(st); run_dir=out.get('run_dir','')
    if not run_dir: return out
    last=read_last_csv_row(os.path.join(run_dir,'logs','summary.csv'))
    for key in ['env','seed','variant','node_method']:
        if out.get(key) in {None,'', 'null'} and last.get(key) not in {None,''}:
            out[key]=last.get(key)
    return out

def _expand_sweep(sweep_path: str, overrides: Optional[List[str]] = None) -> List[Dict]:
    sweep=load_json(sweep_path); base_config=sweep.get('base_config')
    if base_config is None: raise ValueError('Sweep JSON must contain base_config')
    if not os.path.isabs(base_config): base_config=os.path.join(os.path.dirname(sweep_path),base_config)
    tasks=[]
    if 'tasks' in sweep: tasks.extend(sweep['tasks'])
    else:
        grid=sweep.get('grid',{}); envs=grid.get('env',grid.get('envs',[None])); seeds=grid.get('seed',grid.get('seeds',[None])); variants=grid.get('variant',grid.get('variants',[None])); node_methods=grid.get('node_method',grid.get('node_methods',[None]))
        for env,seed,variant,node_method in itertools.product(envs,seeds,variants,node_methods):
            t={}
            if env is not None: t['env']=env
            if seed is not None: t['seed']=seed
            if variant is not None: t['variant']=variant
            if node_method is not None: t['node_method']=node_method
            if 'set' in grid: t['set']=dict(grid['set'])
            tasks.append(t)
    resources=sweep.get('resources',{}); out=[]
    for idx,t in enumerate(tasks):
        env=t.get('env','env'); seed=t.get('seed',0); variant=t.get('variant','full_bars'); node=t.get('node_method','bars'); stamp=time.strftime('%Y%m%d_%H%M%S'); run_id=t.get('run_id',f'{env}_{variant}_{node}_seed{seed}_{idx}_{stamp}'.replace('/','_'))
        set_items=[f'{k}={json.dumps(v) if not isinstance(v,str) else v}' for k,v in (t.get('set',{}) or {}).items()] + list(overrides or [])
        out.append({'run_id':run_id,'base_config':base_config,'env':t.get('env'),'seed':t.get('seed'),'variant':t.get('variant'),'node_method':t.get('node_method'),'set':set_items,'mem_mb':int(t.get('mem_mb',resources.get('default_mem_mb',6000)))})
    return out

def _build_cmd(task: Dict, log_root: str):
    env=task.get('env') or 'unknown_env'; variant=task.get('variant') or 'variant'; run_dir=os.path.join(log_root,str(env),str(variant),str(task['run_id'])); cmd=[sys.executable,'-m','bars.cli','run','--config',task['base_config'],'--run-dir',run_dir]
    if task.get('env') is not None: cmd += ['--env',str(task['env'])]
    if task.get('seed') is not None: cmd += ['--seed',str(task['seed'])]
    if task.get('variant') is not None: cmd += ['--variant',str(task['variant'])]
    if task.get('node_method') is not None: cmd += ['--node-method',str(task['node_method'])]
    for s in task.get('set',[]): cmd += ['--set',s]
    return cmd, run_dir

def _load_all_states(log_root: str) -> List[Dict]:
    states=[]
    for p in sorted(_jobs_dir(log_root).glob('*.json')):
        try: states.append(_read_state(p))
        except Exception: pass
    return states

def _parse_int_map(raw: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not raw:
        return out
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'Expected GPU map item like 0:16, got {item!r}')
        key, value = item.split(':', 1)
        out[int(key.strip())] = int(value.strip())
    return out

def _reconcile_dead_running_states(log_root: str) -> None:
    terminal_status = {'completed', 'failed', 'terminated', 'stopped', 'force_killed'}
    for st in _load_all_states(log_root):
        if st.get('status') in terminal_status:
            continue
        pid = int(st.get('pid', 0) or 0)
        if pid > 0 and _pid_alive(pid):
            continue
        run_dir = st.get('run_dir', '')
        last = read_last_csv_row(os.path.join(run_dir, 'logs', 'summary.csv')) if run_dir else {}
        completed = (
            last.get('status') == 'completed'
            or last.get('phase') == 'completed'
            or _archive_exists(run_dir)
        )
        st.update({
            'status': 'completed' if completed else 'stopped',
            'returncode': 0 if completed else st.get('returncode', ''),
            'ended_at': st.get('ended_at', time.time()),
            'reconciled_at': time.time(),
        })
        _write_state(log_root, st)
        if run_dir:
            try: save_json(st, os.path.join(run_dir, 'job.json'))
            except Exception: pass

def _assign_gpu(
    gpu_indices: Optional[List[int]],
    running: List[Dict],
    max_jobs_per_gpu: int,
    mem_mb: int,
    reserved: Dict[int,int],
    reserve_free_mb: int = 0,
    max_jobs_per_gpu_map: Optional[Dict[int, int]] = None,
) -> Optional[int]:
    infos=query_gpus(gpu_indices)
    if not infos: return 0
    counts={g.index:0 for g in infos}
    for st in running:
        if st.get('status')=='running' and st.get('gpu') in counts: counts[int(st['gpu'])]+=1
    infos.sort(key=lambda g:(counts.get(g.index,0), -(g.memory_free_mb-reserved.get(g.index,0))))
    for g in infos:
        limit = (max_jobs_per_gpu_map or {}).get(g.index, max_jobs_per_gpu)
        effective_free = g.memory_free_mb - reserved.get(g.index, 0)
        if counts[g.index] < limit and effective_free - mem_mb >= reserve_free_mb:
            reserved[g.index]=reserved.get(g.index,0)+mem_mb
            return g.index
    return None

def _launch_one(task: Dict, log_root: str, gpu: int) -> subprocess.Popen:
    cmd,run_dir=_build_cmd(task,log_root); Path(run_dir).mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env.setdefault('D4RL_SUPPRESS_IMPORT_ERROR','1')
    env.setdefault('BARS_DISABLE_TQDM','1')
    env.setdefault('BARS_TQDM_MININTERVAL','10')
    # Avoid CPU oversubscription when many GPU jobs construct SciPy/sklearn graphs concurrently.
    thread_count = str(env.get('BARS_JOB_NUM_THREADS', '1'))
    for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[_k] = thread_count
    stdout=open(os.path.join(run_dir,'stdout.log'),'a',buffering=1,encoding='utf-8'); stderr=open(os.path.join(run_dir,'stderr.log'),'a',buffering=1,encoding='utf-8')
    state={'run_id':task['run_id'],'status':'launching','gpu':gpu,'mem_mb':task['mem_mb'],'run_dir':run_dir,'cmd':cmd,'created_at':time.time(),'env':task.get('env'),'seed':task.get('seed'),'variant':task.get('variant'),'node_method':task.get('node_method')}; save_json(state,os.path.join(run_dir,'job.json')); _write_state(log_root,state)
    proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,preexec_fn=os.setsid); state.update({'status':'running','pid':proc.pid,'pgid':os.getpgid(proc.pid),'started_at':time.time()}); save_json(state,os.path.join(run_dir,'job.json')); _write_state(log_root,state); return proc

def scheduler_loop(args) -> None:
    tasks=_expand_sweep(args.sweep,overrides=args.set)
    if args.min_task_mem_mb > 0:
        raised = 0
        for task in tasks:
            if int(task.get('mem_mb', 0) or 0) < args.min_task_mem_mb:
                task['mem_mb'] = args.min_task_mem_mb
                raised += 1
        if raised:
            print(f'raised mem_mb to min_task_mem_mb={args.min_task_mem_mb} for {raised} tasks', flush=True)
    gpu_indices=parse_gpu_list(args.gpus); gpu_limit_map=_parse_int_map(args.max_jobs_per_gpu_map); existing={str(st.get('run_id')) for st in _load_all_states(args.log_root) if st.get('run_id')}; pending=[t for t in tasks if str(t.get('run_id')) not in existing]; procs={}; _jobs_dir(args.log_root)
    if existing:
        print(f'skipping {len(existing)} existing job states in {args.log_root}', flush=True)
    print(f'scheduler limits max_jobs_per_gpu={args.max_jobs_per_gpu} max_jobs_per_gpu_map={gpu_limit_map} reserve_free_mb={args.reserve_free_mb}', flush=True)
    with open(_scheduler_pid_path(args.log_root),'w',encoding='utf-8') as f: f.write(str(os.getpid()))
    while pending or procs:
        for run_id,proc in list(procs.items()):
            ret=proc.poll()
            if ret is None: continue
            st_path=_state_path(args.log_root,run_id); st=_read_state(st_path) if st_path.exists() else {'run_id':run_id}; prior=st.get('status','running'); status='stopped' if prior=='stop_requested' else ('completed' if ret==0 else 'failed'); st.update({'status':status,'returncode':ret,'ended_at':time.time()}); _write_state(args.log_root,st)
            if st.get('run_dir'):
                try: save_json(st,os.path.join(st['run_dir'],'job.json'))
                except Exception: pass
            print(f'job finished run_id={run_id} status={status} rc={ret} run_dir={st.get("run_dir","")}', flush=True)
            del procs[run_id]
        _reconcile_dead_running_states(args.log_root)
        running_states=_load_all_states(args.log_root); reserved={}; launched=0
        for task in list(pending):
            gpu=_assign_gpu(gpu_indices,running_states,args.max_jobs_per_gpu,task['mem_mb'],reserved,args.reserve_free_mb,gpu_limit_map)
            if gpu is None: continue
            if args.dry_run:
                cmd,run_dir=_build_cmd(task,args.log_root); print('DRY-RUN',gpu,task['mem_mb'],' '.join(cmd),'->',run_dir); pending.remove(task); continue
            proc=_launch_one(task,args.log_root,gpu); procs[task['run_id']]=proc; pending.remove(task); launched+=1
            running_states.append({'status': 'running', 'gpu': gpu})
            print(f'launched run_id={task["run_id"]} gpu={gpu} mem_mb={task["mem_mb"]} env={task.get("env")} seed={task.get("seed")} variant={task.get("variant")} node_method={task.get("node_method")} run_dir={_build_cmd(task,args.log_root)[1]}', flush=True)
            if launched >= max(1,args.launch_burst): break
        if args.dry_run and not pending: break
        time.sleep(float(args.poll_seconds))

def launch_background(args) -> None:
    cmd=[sys.executable,'-m','bars.sched.jobctl','scheduler','--sweep',args.sweep,'--log-root',args.log_root,'--gpus',args.gpus,'--max-jobs-per-gpu',str(args.max_jobs_per_gpu),'--poll-seconds',str(args.poll_seconds),'--launch-burst',str(args.launch_burst)]
    cmd += ['--reserve-free-mb', str(args.reserve_free_mb), '--min-task-mem-mb', str(args.min_task_mem_mb)]
    if args.max_jobs_per_gpu_map: cmd += ['--max-jobs-per-gpu-map', args.max_jobs_per_gpu_map]
    for s in args.set or []: cmd += ['--set',s]
    if args.dry_run: cmd.append('--dry-run')
    _jobs_dir(args.log_root); out=open(_jobs_dir(args.log_root)/'scheduler.out','a',buffering=1,encoding='utf-8'); proc=subprocess.Popen(cmd,stdout=out,stderr=out,preexec_fn=os.setsid)
    with open(_scheduler_pid_path(args.log_root),'w',encoding='utf-8') as f: f.write(str(proc.pid))
    print(f'scheduler started pid={proc.pid} log_root={args.log_root}')

def print_status(args) -> None:
    states=_load_all_states(args.log_root); wanted=parse_gpu_list(args.gpus); state_gpu_ids=sorted({int(st['gpu']) for st in states if st.get('gpu') is not None})
    combined=None if wanted is None else sorted(set(wanted).union(state_gpu_ids))
    infos=query_gpus(combined)
    if infos:
        print('GPUs:')
        for g in infos: print(f'  gpu={g.index} free={g.memory_free_mb}MB total={g.memory_total_mb}MB util={g.utilization_gpu}%')
    pid_path=_scheduler_pid_path(args.log_root)
    if pid_path.exists(): pid=int(pid_path.read_text().strip() or '0'); print(f'scheduler pid={pid} alive={_pid_alive(pid)}')
    print('Jobs:')
    for st in states:
        st=_state_with_summary_fallback(st)
        pid=int(st.get('pid',0) or 0); alive=_pid_alive(pid); run_dir=st.get('run_dir',''); last=read_last_csv_row(os.path.join(run_dir,'logs','summary.csv')) if run_dir else {}; tail=(last.get('phase') or last.get('status') or '') if last else ''; update=_last_log_update(run_dir); archive_exists=_archive_exists(run_dir)
        terminal_status={"completed","failed","terminated","stopped","force_killed"}
        is_stale=bool(update['stale']) and st.get('status') not in terminal_status and alive
        print(f"  run_id={st.get('run_id')} pid={pid} gpu={st.get('gpu')} env={st.get('env')} seed={st.get('seed')} variant={st.get('variant')} node_method={st.get('node_method')} status={st.get('status')} alive={alive} rc={st.get('returncode','')} last_phase={tail} last_log_update_min={'' if update['last_log_update_age_min'] is None else round(float(update['last_log_update_age_min']),1)} stale={int(is_stale)} archive_exists={int(archive_exists)}")
        print(f"    dir={run_dir}")

def stop_jobs(args) -> None:
    states=_load_all_states(args.log_root); targets=[s for s in states if s.get('status') in {'running','launching'}] if args.all else [s for s in states if s.get('run_id')==args.run_id]
    if not targets: print('No matching running jobs.'); return
    for st in targets:
        pid=int(st.get('pid',0) or 0); pgid=int(st.get('pgid',pid) or pid)
        if pid<=0 or not _pid_alive(pid): st['status']='stopped'; _write_state(args.log_root,st); continue
        sig=signal.SIGKILL if args.force else signal.SIGTERM
        try:
            os.killpg(pgid,sig); st['status']='force_killed' if args.force else 'stop_requested'; st['stop_requested_at']=time.time(); _write_state(args.log_root,st)
            if st.get('run_dir'): save_json(st,os.path.join(st['run_dir'],'job.json'))
            print(f'sent {sig.name} to run_id={st.get("run_id")} pgid={pgid}')
        except Exception as exc: print(f'failed to stop {st.get("run_id")}: {exc}')

def pack_jobs(args) -> None:
    states=_load_all_states(args.log_root); targets=states if args.all else [s for s in states if s.get('run_id')==args.run_id]
    for st in targets:
        rd=st.get('run_dir')
        if rd and os.path.isdir(rd): print(package_logs(rd))

def main(argv=None) -> None:
    parser=argparse.ArgumentParser(prog='barsctl'); sub=parser.add_subparsers(dest='cmd',required=True)
    launch=sub.add_parser('launch'); launch.add_argument('--sweep',required=True); launch.add_argument('--log-root',default='runs'); launch.add_argument('--gpus',default='auto'); launch.add_argument('--max-jobs-per-gpu',type=int,default=1); launch.add_argument('--max-jobs-per-gpu-map',default=''); launch.add_argument('--reserve-free-mb',type=int,default=0); launch.add_argument('--min-task-mem-mb',type=int,default=0); launch.add_argument('--poll-seconds',type=float,default=10); launch.add_argument('--launch-burst',type=int,default=1); launch.add_argument('--background',action='store_true'); launch.add_argument('--dry-run',action='store_true'); launch.add_argument('--set',action='append',default=[])
    sched=sub.add_parser('scheduler'); sched.add_argument('--sweep',required=True); sched.add_argument('--log-root',default='runs'); sched.add_argument('--gpus',default='auto'); sched.add_argument('--max-jobs-per-gpu',type=int,default=1); sched.add_argument('--max-jobs-per-gpu-map',default=''); sched.add_argument('--reserve-free-mb',type=int,default=0); sched.add_argument('--min-task-mem-mb',type=int,default=0); sched.add_argument('--poll-seconds',type=float,default=10); sched.add_argument('--launch-burst',type=int,default=1); sched.add_argument('--dry-run',action='store_true'); sched.add_argument('--set',action='append',default=[])
    status=sub.add_parser('status'); status.add_argument('--log-root',default='runs'); status.add_argument('--gpus',default='auto')
    stop=sub.add_parser('stop'); stop.add_argument('--log-root',default='runs'); stop.add_argument('--run-id',default=None); stop.add_argument('--all',action='store_true'); stop.add_argument('--force',action='store_true')
    pack=sub.add_parser('pack'); pack.add_argument('--log-root',default='runs'); pack.add_argument('--run-id',default=None); pack.add_argument('--all',action='store_true')
    args=parser.parse_args(argv)
    if args.cmd=='launch': launch_background(args) if args.background else scheduler_loop(args)
    elif args.cmd=='scheduler': scheduler_loop(args)
    elif args.cmd=='status': print_status(args)
    elif args.cmd=='stop':
        if not args.all and not args.run_id: raise SystemExit('Provide --run-id or --all')
        stop_jobs(args)
    elif args.cmd=='pack':
        if not args.all and not args.run_id: raise SystemExit('Provide --run-id or --all')
        pack_jobs(args)
if __name__=='__main__': main()

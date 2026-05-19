# Stage23 Protocol Audit

## Official GAS Repo
- Path: `external_src/GAS`
- Commit: `c9e590fcd6f082de677d332a84e44a1a631da5c5`
- Dirty: `True`
```
M K_utils/graph_builder.py
 M O_utils/log_utils.py
 M construct_graph.py
 M evaluate_gas.py
 M pretrain_tdr.py
 M train_policy.py
?? D_utils/__pycache__/
?? K_utils/__pycache__/
?? M_utils/__pycache__/
?? M_utils/agents/__pycache__/
?? O_utils/__pycache__/
?? artifacts/
```
```
origin	https://github.com/qortmdgh4141/GAS (fetch)
origin	https://github.com/qortmdgh4141/GAS (push)
```

## Environment Checks
```csv
env,seed,artifact_complete,policy_checkpoint_step,tdr_checkpoint_step,keygraph_node_count,keygraph_edge_count,keygraph_way_steps,task_id_count,max_episode_steps,goal_present_all_checked,protocol_status
antmaze-medium-navigate-v0,0,True,100000,100000,412,5280,8.0,5,1000,True,ok
antmaze-medium-stitch-v0,0,True,100000,100000,485,10166,8.0,5,1000,True,ok
```

## Red Flags
- Local `external_src/GAS` is dirty; A-route reproduction must distinguish pristine official from locally patched official scripts.
- `antmaze-medium-navigate-v0` uses policy checkpoint step `100000`, below full official 1M/500k reproduction.
- `antmaze-medium-stitch-v0` uses policy checkpoint step `100000`, below full official 1M/500k reproduction.

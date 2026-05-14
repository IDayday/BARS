# Route-B official source audit

This audit checks whether official HIQL/GAS repositories are present and records the exact commit/file fingerprints used for strong-backbone alignment. It also lists intentional BARS-vs-official differences.

## Repositories
- HIQL: `external_src/HIQL` exists=True head=`b32c832cd67e4fe56ecabb49dbb96dc8d4f4c432`
- GAS: `external_src/GAS` exists=True head=`c9e590fcd6f082de677d332a84e44a1a631da5c5`

## GAS files
- `external_src/GAS/construct_graph.py` exists=True sha256=ec6fd3348028f2f1247b2f945fbee65694134d546e45948c26bfc67f2b09f1c3
  - pattern_check: {'Temporal': False, 'efficiency': False, 'keygraph': True, 'NetworkX': False, 'way_steps': True, 'H_TD': False}
- `external_src/GAS/pretrain_tdr.py` exists=True sha256=23dd431b51dd34a6d249aa10c66e33dc7c17cbfe605666c9d4bc6d484e7f7a42
  - pattern_check: {'Temporal': False, 'efficiency': False, 'keygraph': False, 'NetworkX': False, 'way_steps': False, 'H_TD': False}
- `external_src/GAS/train_policy.py` exists=True sha256=a14ca7f516c8cfabc45baa689a390b763dd8ef629ec95a50daae21c0ff23b0be
  - pattern_check: {'Temporal': False, 'efficiency': False, 'keygraph': False, 'NetworkX': False, 'way_steps': False, 'H_TD': False}
- `external_src/GAS/evaluate_gas.py` exists=True sha256=63eccd6e54d50f2b29e58fc6b456df1e5cf0349d9f30e5822eb40c255eaec4e4
  - pattern_check: {'Temporal': False, 'efficiency': False, 'keygraph': True, 'NetworkX': False, 'way_steps': True, 'H_TD': False}
- `external_src/GAS/K_utils/keygraph_utils.py` exists=True sha256=e333444a867d5d7d34c3fa04986b64c61eadc491f2cde1bc214e95994c92ff91
  - pattern_check: {'Temporal': False, 'efficiency': False, 'keygraph': True, 'NetworkX': False, 'way_steps': True, 'H_TD': False}
- `external_src/GAS/K_utils/graph_builder.py` exists=True sha256=7bbbc40f6352e742c07d549f70d5f2866eea675240a460169c0932b944de84fc
  - pattern_check: {'Temporal': False, 'efficiency': False, 'keygraph': True, 'NetworkX': False, 'way_steps': True, 'H_TD': False}

## HIQL files
- `external_src/HIQL/README.md` exists=True sha256=4aa0579e8fde7eaaef5c4eddba857b6bf6dfac689599792bc45a7d59c3d4dfbd
  - pattern_check: {'hiql': True, 'sample_actions': False, 'low': True, 'high': True, 'antmaze': True, 'way_steps': True, 'use_rep': True}
- `external_src/HIQL/main.py` exists=True sha256=894cd2929d60fb1677a92922e713b679876742f57363d4466e7c4f5b00010ab6
  - pattern_check: {'hiql': True, 'sample_actions': True, 'low': False, 'high': True, 'antmaze': True, 'way_steps': True, 'use_rep': True}
- `external_src/HIQL/src/agents/hiql.py` exists=True sha256=f9c5dc5fbf96db3a3ee052cae0f331f534859f4a19a761fc36dd013a4053ed69
  - pattern_check: {'hiql': False, 'sample_actions': True, 'low': True, 'high': True, 'antmaze': False, 'way_steps': True, 'use_rep': True}
- `external_src/HIQL/src/agents/iql.py` exists=True sha256=d9719c6e682b1d42da895f356621d8d28257dad86b915061c6644725ad33b10e
  - pattern_check: {'hiql': False, 'sample_actions': False, 'low': False, 'high': False, 'antmaze': False, 'way_steps': False, 'use_rep': False}
- `external_src/HIQL/jaxrl_m/evaluation.py` exists=True sha256=ce806c32ddb1aec888b9bba6b4e42e7544b0b14ee1b92b33ea66c552e76c01f2
  - pattern_check: {'hiql': False, 'sample_actions': False, 'low': True, 'high': True, 'antmaze': True, 'way_steps': False, 'use_rep': True}

## Intentional differences / required exact-artifact paths

1. `graph.node_method=gas_te` implements GAS-style TE filtering and TD-aware clustering inside BARS, but maps latent centers back to concrete dataset indices. This is necessary because BARS low-level execution consumes full observations as subgoals.
2. Exact GAS graph comparison should use `external_gas.keygraph_path` plus either `external_gas.node_indices_path` or `external_gas.dataset_embeddings_path`. Without the official embedding space, mapping keygraph centers with BARS embeddings is only an approximation and is logged as such.
3. HIQL exact reproduction is not reimplemented in PyTorch. Use official HIQL repo/checkpoints and connect them through `policy.type=external` and `external_policy.factory`. This avoids silently dropping JAX/Flax implementation details required for SOTA.
4. To claim same-backbone improvement, train/evaluate official GAS or HIQL first, then replace only the planner with BARS shortest/reachability/full_bars in the same environment, same seed, same low-level policy, and same graph where applicable.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Iterable


def sha(path: Path) -> str:
    if not path.exists():
        return 'MISSING'
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception as exc:
        return f'UNKNOWN:{type(exc).__name__}'


def contains(path: Path, patterns: Iterable[str]) -> dict[str, bool]:
    txt = path.read_text(errors='ignore') if path.exists() else ''
    return {p: (p in txt) for p in patterns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hiql-repo', default='external_src/HIQL')
    ap.add_argument('--gas-repo', default='external_src/GAS')
    ap.add_argument('--out', default='reports/routeb_source_audit.md')
    args = ap.parse_args()
    hiql = Path(args.hiql_repo)
    gas = Path(args.gas_repo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append('# Route-B official source audit')
    lines.append('')
    lines.append('This audit checks whether official HIQL/GAS repositories are present and records the exact commit/file fingerprints used for strong-backbone alignment. It also lists intentional BARS-vs-official differences.')
    lines.append('')

    lines.append('## Repositories')
    for name, repo in [('HIQL', hiql), ('GAS', gas)]:
        lines.append(f'- {name}: `{repo}` exists={repo.exists()} head=`{git_head(repo) if repo.exists() else "MISSING"}`')
    lines.append('')

    lines.append('## GAS files')
    gas_files = [
        gas / 'construct_graph.py',
        gas / 'pretrain_tdr.py',
        gas / 'train_policy.py',
        gas / 'evaluate_gas.py',
        gas / 'K_utils' / 'keygraph_utils.py',
        gas / 'K_utils' / 'graph_builder.py',
    ]
    for f in gas_files:
        lines.append(f'- `{f}` exists={f.exists()} sha256={sha(f)}')
        if f.exists():
            pats = contains(f, ['Temporal', 'efficiency', 'keygraph', 'NetworkX', 'way_steps', 'H_TD'])
            lines.append(f'  - pattern_check: {pats}')
    lines.append('')

    lines.append('## HIQL files')
    hiql_files = [
        hiql / 'README.md',
        hiql / 'main.py',
        hiql / 'src' / 'agents' / 'hiql.py',
        hiql / 'src' / 'agents' / 'iql.py',
        hiql / 'jaxrl_m' / 'evaluation.py',
    ]
    for f in hiql_files:
        lines.append(f'- `{f}` exists={f.exists()} sha256={sha(f)}')
        if f.exists():
            pats = contains(f, ['hiql', 'sample_actions', 'low', 'high', 'antmaze', 'way_steps', 'use_rep'])
            lines.append(f'  - pattern_check: {pats}')
    lines.append('')

    lines.append('## Intentional differences / required exact-artifact paths')
    lines.append('')
    lines.append('1. `graph.node_method=gas_te` implements GAS-style TE filtering and TD-aware clustering inside BARS, but maps latent centers back to concrete dataset indices. This is necessary because BARS low-level execution consumes full observations as subgoals.')
    lines.append('2. Exact GAS graph comparison should use `external_gas.keygraph_path` plus either `external_gas.node_indices_path` or `external_gas.dataset_embeddings_path`. Without the official embedding space, mapping keygraph centers with BARS embeddings is only an approximation and is logged as such.')
    lines.append('3. HIQL exact reproduction is not reimplemented in PyTorch. Use official HIQL repo/checkpoints and connect them through `policy.type=external` and `external_policy.factory`. This avoids silently dropping JAX/Flax implementation details required for SOTA.')
    lines.append('4. To claim same-backbone improvement, train/evaluate official GAS or HIQL first, then replace only the planner with BARS shortest/reachability/full_bars in the same environment, same seed, same low-level policy, and same graph where applicable.')

    out.write_text('\n'.join(lines) + '\n')
    print(out)


if __name__ == '__main__':
    main()

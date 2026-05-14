#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_template(backend: str, env_name: str, seed: int) -> dict:
    backend = backend.lower()
    if backend not in {"hiql", "gas"}:
        raise ValueError(f"Unsupported backend {backend!r}; expected hiql or gas")
    return {
        "backend": backend,
        "status": "template",
        "env_name": env_name,
        "seed": seed,
        "adapter_contract": {
            "entrypoint": "Provide a Python wrapper with act(obs_np, goal_np, obs_normalizer, action_low=None, action_high=None, device='cuda') -> np.ndarray",
            "observation_space": "Use full AntMaze observation; overwrite goal xy in the goal observation before calling the backbone.",
            "normalization": "Backbone wrapper is responsible for any extra normalization beyond BARS obs_normalizer.",
            "action_space": "Return unclipped float32 actions; BARS may clip to env bounds.",
        },
        "expected_artifacts": {
            "checkpoint_path": f"external_backbones/{backend}/{env_name}/seed{seed}/checkpoint.pt",
            "config_path": f"external_backbones/{backend}/{env_name}/seed{seed}/config.json",
            "wrapper_module": f"external_backbones/{backend}/adapter.py",
        },
        "notes": [
            "This template is intentionally decoupled from the current GCBC policy path so Route B work does not block Route A ablations.",
            "Populate the wrapper module first, then wire it into BARS once checkpoint loading and action decoding are stable.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hiql", "gas"])
    ap.add_argument("--env", default="antmaze-medium-play-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out or f"configs/route_b/{args.backend}_{args.env.replace('/', '_')}_seed{args.seed}_adapter_template.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_template(args.backend, args.env, args.seed), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

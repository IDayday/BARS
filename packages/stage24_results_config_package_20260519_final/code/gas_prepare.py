from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .gas_artifacts import resolve_gas_artifacts
from .gas_backbone import GASBackbone


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--artifact-root", default="artifacts/gas")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--gpu", default="0")
    p.add_argument("--quick", type=int, default=1)
    p.add_argument("--prefer-pretrained", type=int, default=1)
    p.add_argument("--train-if-missing", type=int, default=1)
    p.add_argument("--export-embeddings", type=int, default=1)
    args = p.parse_args(argv)
    bb = GASBackbone.load_or_train(
        args.env,
        args.seed,
        args.artifact_root,
        args.gas_repo_path,
        args.gpu,
        prefer_pretrained=bool(args.prefer_pretrained),
        train_if_missing=bool(args.train_if_missing),
        quick=bool(args.quick),
    )
    artifacts = bb.artifacts or resolve_gas_artifacts(args.env, args.seed, args.artifact_root)
    if args.export_embeddings and artifacts.dataset_embeddings is None and artifacts.complete:
        bb.export_dataset_embeddings(artifacts.features_dir / "dataset_embeddings.npy")
        artifacts = resolve_gas_artifacts(args.env, args.seed, args.artifact_root)
    if not artifacts.complete:
        raise RuntimeError(f"Incomplete GAS artifacts for {args.env} seed {args.seed}: {artifacts.to_dict()}")
    print(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations
import os, tarfile, time
from pathlib import Path
from typing import Iterable, Optional

def package_logs(run_dir: str, include_patterns: Optional[Iterable[str]] = None) -> str:
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=True); archive_dir = run / 'archives'; archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f'logs_{run.name}_{int(time.time())}.tar.gz'
    patterns = list(include_patterns or ['logs', 'config.json', 'manifest.json', 'job.json'])
    with tarfile.open(archive_path, 'w:gz') as tar:
        for pat in patterns:
            p = run / pat
            if p.exists(): tar.add(str(p), arcname=str(p.relative_to(run)))
    return str(archive_path)

def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: f.write(text)

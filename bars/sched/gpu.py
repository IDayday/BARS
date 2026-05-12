from __future__ import annotations
import csv, subprocess
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List, Optional
@dataclass
class GPUInfo:
    index: int; memory_free_mb: int; memory_total_mb: int; utilization_gpu: int = -1
def query_gpus(indices: Optional[Iterable[int]] = None) -> List[GPUInfo]:
    cmd=['nvidia-smi','--query-gpu=index,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits']
    try: out=subprocess.check_output(cmd,text=True,stderr=subprocess.DEVNULL)
    except Exception: return []
    wanted=None if indices is None else {int(i) for i in indices}; infos=[]
    for row in csv.reader(StringIO(out)):
        if len(row)<3: continue
        idx=int(row[0].strip())
        if wanted is not None and idx not in wanted: continue
        infos.append(GPUInfo(idx,int(row[1].strip()),int(row[2].strip()),int(row[3].strip()) if len(row)>3 and row[3].strip().isdigit() else -1))
    return infos
def parse_gpu_list(raw: str) -> Optional[List[int]]:
    if raw.lower() in {'auto','all',''}: return None
    return [int(x) for x in raw.split(',') if x.strip()]

#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--log-root',default='runs'); ap.add_argument('--out-dir',default=None); args=ap.parse_args(); root=Path(args.log_root); out=Path(args.out_dir) if args.out_dir else root/'_analysis'; out.mkdir(parents=True,exist_ok=True); buckets={}
    for csv_path in root.glob('**/logs/*.csv'):
        try: df=pd.read_csv(csv_path)
        except Exception: continue
        df['run_dir']=str(csv_path.parents[1]); buckets.setdefault(csv_path.name,[]).append(df)
    for name,frames in buckets.items():
        merged=pd.concat(frames,ignore_index=True,sort=False); out_path=out/name.replace('.csv','_all.csv'); merged.to_csv(out_path,index=False); print(out_path,len(merged))
if __name__=='__main__': main()

#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

CONFIG="${CONFIG:-configs/stage23_d4rl_protocol_repair.json}"
STAGE23_ROOT="${STAGE23_ROOT:-runs_stage23_d4rl}"
LOG_ROOT="${LOG_ROOT:-runs_stage23_d4rl_logs}"
QUICK="${QUICK:-1}"

CONFIG="$CONFIG" STAGE23_ROOT="$STAGE23_ROOT" LOG_ROOT="$LOG_ROOT" QUICK="$QUICK" bash scripts/stage23_run_key_claim.sh || true

python scripts/analyze_stage23.py --eval-root "$STAGE23_ROOT" --artifact-root artifacts/stage22 --out reports || true

python - <<'PY'
from pathlib import Path
import pandas as pd
root=Path("runs_stage23_d4rl")
frames=[]
for p in root.rglob("eval.csv"):
    try:
        df=pd.read_csv(p); df["path"]=str(p); frames.append(df)
    except Exception:
        pass
out=Path("reports/stage23_d4rl_protocol_report.md")
out.parent.mkdir(parents=True, exist_ok=True)
if not frames:
    out.write_text("# Stage23 D4RL Protocol Report\n\nNo D4RL eval rows found. REPAIR_D4RL_BACKBONE.\n")
else:
    df=pd.concat(frames, ignore_index=True)
    grouped=df.groupby(["env","variant","fallback_mode"], dropna=False).agg(episodes=("success","count"), success=("success","mean")).reset_index()
    decision="GO_D4RL_EXPANSION" if (grouped[(grouped.variant=="gas_shortest")]["success"].max() if len(grouped[(grouped.variant=="gas_shortest")]) else 0) > 0.2 else "REPAIR_D4RL_BACKBONE"
    try:
        table=grouped.to_markdown(index=False)
    except ImportError:
        table="```csv\n"+grouped.to_csv(index=False).strip()+"\n```"
    out.write_text("# Stage23 D4RL Protocol Report\n\n"+table+"\n\nDecision: "+decision+"\n")
PY

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${ROOT:-external_src}"
mkdir -p "$ROOT" reports
if [ ! -d "$ROOT/HIQL/.git" ]; then
  git clone https://github.com/seohongpark/HIQL.git "$ROOT/HIQL"
else
  git -C "$ROOT/HIQL" fetch --all --tags || true
fi
if [ ! -d "$ROOT/GAS/.git" ]; then
  git clone https://github.com/qortmdgh4141/GAS.git "$ROOT/GAS"
else
  git -C "$ROOT/GAS" fetch --all --tags || true
fi
{
  echo "# Route-B external source status"
  echo
  echo "## HIQL"
  git -C "$ROOT/HIQL" rev-parse HEAD || true
  git -C "$ROOT/HIQL" status --short || true
  echo
  echo "## GAS"
  git -C "$ROOT/GAS" rev-parse HEAD || true
  git -C "$ROOT/GAS" status --short || true
} | tee reports/routeb_external_source_status.md
python scripts/audit_routeb_sources.py --hiql-repo "$ROOT/HIQL" --gas-repo "$ROOT/GAS" --out reports/routeb_source_audit.md || true

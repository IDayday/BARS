#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p reports
branch="$(git branch --show-current)"
commit="$(git rev-parse HEAD)"
logline="$(git log -1 --oneline)"
status_short="$(git status --short || true)"
cat > reports/tmd_test_branch_audit.md <<EOF
# tmd-test Branch Audit

## Git state
- original required branch: stage25
- user-authorized branch for this run: ${branch}
- commit: ${commit}
- latest commit: ${logline}
- dirty state:

\`\`\`text
${status_short:-clean}
\`\`\`

## Directory overview

\`\`\`text
$(find . -maxdepth 3 -type d | sort | sed -n '1,160p')
\`\`\`

## Existing GAS/BARS/TMD-related files

\`\`\`text
$(find scripts configs reports bars external_src -maxdepth 4 -type f 2>/dev/null | sort | grep -Ei 'gas|bars|tmd|stage' | sed -n '1,320p')
\`\`\`

## Existing scripts/configs/reports that can be reused

- bars/external and bars/gas_bars contain existing GAS/BARS adapters and evaluators.
- external_src/GAS is the vendored GAS implementation.
- external_src/tmd-release is the official TMD release cloned for this run.
- /mnt/project/offlinerl_datasets/ogbench is the local OGBench dataset source.

## Files that must not be used as evidence
- main-branch reports/results are stale and excluded.

## Planned tmd-test additions

- bars/tmd_test/*
- configs/tmd_test/*
- scripts/tmd_test_*
- reports/tmd_test_*
- artifacts/tmd_test/*
- runs_tmd_test/*
EOF

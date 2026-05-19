#!/usr/bin/env bash
set -euo pipefail

GAS_REPO_URL="${GAS_REPO_URL:-https://github.com/qortmdgh4141/GAS.git}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
GAS_REF="${GAS_REF:-main}"
PATCH_FILE="${PATCH_FILE:-third_party/gas_stage22.patch}"

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
patch_path="$repo_root/$PATCH_FILE"

if [[ ! -f "$patch_path" ]]; then
  echo "[setup_gas_repo] Missing patch file: $patch_path" >&2
  exit 2
fi

if [[ -d "$GAS_REPO_PATH/.git" ]]; then
  echo "[setup_gas_repo] Reusing existing GAS checkout at $GAS_REPO_PATH"
  apply_mode="nested_git"
elif [[ -d "$GAS_REPO_PATH" ]]; then
  echo "[setup_gas_repo] Using vendored GAS source at $GAS_REPO_PATH"
  apply_mode="vendored"
else
  mkdir -p "$(dirname "$GAS_REPO_PATH")"
  git clone "$GAS_REPO_URL" "$GAS_REPO_PATH"
  git -C "$GAS_REPO_PATH" checkout "$GAS_REF"
  apply_mode="nested_git"
fi

apply_patch() {
  if [[ "$apply_mode" == "nested_git" ]]; then
    git -C "$GAS_REPO_PATH" apply --unidiff-zero --ignore-space-change "$@" "$patch_path"
  else
    git -C "$repo_root" apply --unidiff-zero --ignore-space-change --directory="$GAS_REPO_PATH" "$@" "$patch_path"
  fi
}

if apply_patch --check >/dev/null 2>&1; then
  apply_patch
  echo "[setup_gas_repo] Applied Stage22/Stage23 GAS compatibility patch."
elif apply_patch --reverse --check >/dev/null 2>&1; then
  echo "[setup_gas_repo] Stage22/Stage23 GAS compatibility patch is already applied."
else
  apply_patch --check || true
  echo "[setup_gas_repo] Patch does not apply cleanly. Inspect the GAS source at $GAS_REPO_PATH." >&2
  exit 1
fi

if [[ "$apply_mode" == "nested_git" ]]; then
  git -C "$GAS_REPO_PATH" status -sb
else
  git -C "$repo_root" status -sb -- "$GAS_REPO_PATH"
fi

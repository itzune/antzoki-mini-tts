#!/usr/bin/env bash
# =============================================================================
#  Upload hf_release/<name>/ to the Hugging Face Hub. PRIVATE by default.
#
#  Prerequisite (once):  source scripts/env.sh && hf auth login   (token with "write" permission)
#  Usage:   bash scripts/upload_hf.sh <user>/<repo> [hf_release/<repo>]
#           PUBLIC=1 bash scripts/upload_hf.sh ...      # create it as public
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" >/dev/null

REPO_ID="${1:?usage: upload_hf.sh <user>/<repo> [folder]}"
FOLDER="${2:-$ROOT/hf_release/${REPO_ID##*/}}"
[[ -f "$FOLDER/model.safetensors" && -f "$FOLDER/README.md" ]] || {
  echo "Missing $FOLDER/model.safetensors or README.md; run scripts/export_hf.py first" >&2; exit 2; }

hf auth whoami >/dev/null 2>&1 || { echo "Not logged in: run 'hf auth login'" >&2; exit 2; }

# Safety: never upload training/optimizer files or local paths
if find "$FOLDER" \( -name 'training_args.bin' -o -name '*optim*' -o -name 'global_step*' \) | grep -q .; then
  echo "The folder contains training files; aborting." >&2; exit 2
fi
if grep -rIl "/home/" "$FOLDER" --include='*.json' --include='*.md' --include='*.py' | grep -q .; then
  echo "Local paths found in:" >&2; grep -rIl "/home/" "$FOLDER" --include='*.json' --include='*.md' --include='*.py' >&2; exit 2
fi

VIS="--private"; [[ "${PUBLIC:-0}" == "1" ]] && VIS=""
echo "[hf] $FOLDER → https://huggingface.co/$REPO_ID  (${VIS:-public})"
du -sh "$FOLDER"
hf repo create "$REPO_ID" --repo-type model $VIS --exist-ok
hf upload "$REPO_ID" "$FOLDER" . --repo-type model --commit-message "Upload Basque fine-tune of Audio8-TTS-Realistic-Air"
echo "[hf] done: https://huggingface.co/$REPO_ID"

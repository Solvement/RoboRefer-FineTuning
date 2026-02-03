#!/usr/bin/env bash
#
# Sync RoboRefer repo FROM remote server TO local current directory.
# Run this script on your LOCAL machine (not on the server).
#
# Keeps only: source code (.py, .cpp, .h), configs (.yaml, .json), scripts (.sh), docs (.md).
# Excludes: large data, model weights, cache, venvs, system files.
#

set -e

# --- Configure these (run from local machine) ---
REMOTE_USER="${REMOTE_USER:-ky2738}"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST, e.g. export REMOTE_HOST=login.cluster.edu}"
REMOTE_PATH="${REMOTE_PATH:-/local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer}"
LOCAL_DEST="${1:-.}"

# Remote source (no trailing slash so we sync contents into LOCAL_DEST)
REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

echo "=========================================="
echo "Sync: Remote -> Local"
echo "  From: ${REMOTE}"
echo "  To:   ${LOCAL_DEST}"
echo "=========================================="
echo ""
echo "Excluding: data/, datasets/, scannet/, runs/, tmp/, venv/, .conda/, env/,"
echo "            __pycache__/, *.pyc, *.pth, *.ckpt, *.safetensors, *.weights, .DS_Store"
echo "Including: *.py, *.cpp, *.h, *.yaml, *.json, *.sh, *.md"
echo ""

# rsync: first exclude dirs we never want to descend into, then include only wanted file types.
# Order matters: excludes first, then include '*/' to descend, then include patterns, then exclude '*'.
rsync -avz --progress \
  --exclude='data/' \
  --exclude='datasets/' \
  --exclude='scannet/' \
  --exclude='runs/' \
  --exclude='tmp/' \
  --exclude='venv/' \
  --exclude='.conda/' \
  --exclude='env/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.weights' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --include='*/' \
  --include='*.py' \
  --include='*.cpp' \
  --include='*.h' \
  --include='*.yaml' \
  --include='*.json' \
  --include='*.sh' \
  --include='*.md' \
  --exclude='*' \
  "$REMOTE" \
  "$LOCAL_DEST"

echo ""
echo "=========================================="
echo "Done. Only .py, .cpp, .h, .yaml, .json, .sh, .md were copied."
echo "=========================================="

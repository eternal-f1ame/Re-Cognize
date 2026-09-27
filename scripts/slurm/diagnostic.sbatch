#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#
# One diagnostic script on one GPU. The analysis scripts under analysis/ run on
# CPU for a few series; a full test-set sweep needs a GPU and a queue slot.
#   DIAG_CMD="analysis/score_fusion.py --device cuda" \
#     sbatch --partition=short --gres=gpu:turing:1 --exclude=$(...) scripts/slurm/diagnostic.sbatch
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

DIAG_CMD="${DIAG_CMD:?DIAG_CMD must hold the script and its arguments}"
echo "[diag] $DIAG_CMD host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[diag] no GPU visible"
exec "$PY" $DIAG_CMD

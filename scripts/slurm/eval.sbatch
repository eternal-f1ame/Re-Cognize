#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#
# Evaluation array: one manifest row per array task.
#   bash scripts/slurm/submit_eval.sh --groups popcharacters
# Resumable: --reuse-cached skips series that already have a valid JSON.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

MANIFEST="${EVAL_MANIFEST:-results/eval_manifest.tsv}"
IDX="${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID}"
LINE="$(sed -n "$((IDX + 1))p" "$MANIFEST")"
[ -n "$LINE" ] || { echo "no manifest row $IDX in $MANIFEST"; exit 2; }
IFS=$'\t' read -r GROUP TAG KIND ARG BACKBONE BATCH CORPUS EXTRA <<< "$LINE"

if [ "$KIND" = checkpoint ]; then SRC=(--checkpoint "$ARG"); else SRC=(--pretrained "$ARG"); fi
OUT="results/$CORPUS/$TAG"
echo "[eval] $GROUP $TAG ($CORPUS) host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[eval] no GPU visible"
exec "$PY" scripts/evaluate.py "${SRC[@]}" --dataset "$CORPUS" --split test \
    --batch-size "$BATCH" --num-workers 4 --reuse-cached --out "$OUT" $EXTRA

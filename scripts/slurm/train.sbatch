#!/bin/bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=checkpoints/%x_%A_%a.out
#SBATCH --error=checkpoints/%x_%A_%a.err
#
# Training array: one line of $RUN_LIST per array task.
#   python scripts/train.py sbatch                       # full campaign on `normal`
#   python scripts/train.py sbatch --smoke --partition short --runs ...
# Runs are resumable (--resume), so a preempted task continues from its last epoch checkpoint.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

RUN_LIST="${RUN_LIST:?RUN_LIST must point at the run-name list}"
IDX="${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID}"
NAME="$(sed -n "$((IDX + 1))p" "$RUN_LIST")"
[ -n "$NAME" ] || { echo "no run at index $IDX in $RUN_LIST"; exit 2; }
MODE_FLAG=""
case "${RUN_MODE:-}" in smoke) MODE_FLAG="--smoke";; timing) MODE_FLAG="--timing";; esac
# The array may come from a manifest other than the campaign's; the run name only resolves there.
MANIFEST_FLAG=()
[ -n "${RUN_MANIFEST:-}" ] && MANIFEST_FLAG=(--manifest "$RUN_MANIFEST")

echo "[train] $NAME (${RUN_MODE:-full}) manifest=${RUN_MANIFEST:-configs/runs.yaml} host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?) submitted_at=${SUBMIT_COMMIT:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[train] no GPU visible"
exec "$PY" scripts/train.py "${MANIFEST_FLAG[@]}" run "$NAME" $MODE_FLAG

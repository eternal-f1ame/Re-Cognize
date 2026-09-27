#!/bin/bash
# Build the evaluation manifest and submit it as a SLURM array.
#   bash scripts/slurm/submit_eval.sh --groups popcharacters
#   DRY_RUN=1 bash scripts/slurm/submit_eval.sh --groups popcharacters manga109
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
STAMP=$(date +%Y%m%d-%H%M%S)
MANIFEST="results/eval_manifest_${STAMP}.tsv"
mkdir -p results/logs
"$PY" scripts/slurm/make_eval_manifest.py "$@" --out "$MANIFEST" | tail -1
N=$(wc -l < "$MANIFEST")
if [ "$N" -eq 0 ]; then
  echo "0 evaluation jobs: nothing to submit"      # not an error: everything is already evaluated
  rm -f "$MANIFEST"
  exit 0
fi
PASCAL=$(sinfo -p short,normal -N -h -o "%N %G" | awk '$2 ~ /pascal/ {print $1}' | sort -u | paste -sd, -)
EXCL=(); [ -n "$PASCAL" ] && EXCL=(--exclude="$PASCAL")
CMD=(sbatch --partition=short --array="0-$((N - 1))" --gres=gpu:turing:1 "${EXCL[@]}" scripts/slurm/eval.sbatch)
echo "${CMD[*]}"
echo "EVAL_MANIFEST=$MANIFEST"
[ "${DRY_RUN:-0}" = 1 ] && exit 0
EVAL_MANIFEST="$MANIFEST" "${CMD[@]}"

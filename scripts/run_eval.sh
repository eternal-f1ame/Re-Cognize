#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem16
#SBATCH --time=100:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Evaluation launcher (thin wrapper around scripts/evaluate.py).
#
# Usage (direct):
#   ./scripts/run_eval.sh --checkpoint checkpoints/magiv2/memory/seed0/final.pth \
#       --out results/popcharacters/magiv2_memory_seed0
#   ./scripts/run_eval.sh --pretrained magiv2 --split test
#   ./scripts/run_eval.sh --checkpoint ... --out ... --series Bakuman --protocols p1 p3 --seeds 0
#
# Usage (SLURM):
#   sbatch --job-name=eval-magiv2 scripts/run_eval.sh --checkpoint checkpoints/magiv2/memory/seed0/final.pth \
#       --out results/popcharacters/magiv2_memory_seed0
#
# PYTHONHASHSEED is pinned here because evaluate.py refuses to run without it.

set -e
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONPATH="$(dirname "$0")/../src:$(dirname "$0")/..:${PYTHONPATH}"
exec python scripts/evaluate.py "$@"

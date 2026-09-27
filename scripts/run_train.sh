#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=100:00:00
#SBATCH --output=checkpoints/%x_%j.out
#SBATCH --error=checkpoints/%x_%j.err
#
# Training launcher (thin wrapper around scripts/train.py).
#
#   ./scripts/run_train.sh list
#   ./scripts/run_train.sh run transreid_memory_seed0
#   ./scripts/run_train.sh sbatch --groups grid            # submit the campaign
#   ./scripts/run_train.sh sbatch --smoke --partition short --runs transreid_memory_seed0
#
# PYTHONHASHSEED is pinned here because training refuses to start without it.

set -e
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONPATH="$(dirname "$0")/../src:$(dirname "$0")/..:${PYTHONPATH}"
exec python scripts/train.py "$@"

#!/bin/bash
# Computes the attention data for the page: sbatch run.slurm (from the repo root, with the repo's
# .venv set up by `uv sync`). Writes the JSON files for the page and the per-head binaries.
#SBATCH --job-name=hos-attn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')")/../.."   # repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi -L
OUT=${OUT:-$PWD/website/src/assets/attention}
.venv/bin/python figure-src/attention/extract_attention.py --out "$OUT" --full-out "${FULL_OUT:-$HOME/hos-attention-full}"
mkdir -p website/public/attention && mv "$OUT"/*.heads.bin website/public/attention/
ls -la "$OUT" website/public/attention

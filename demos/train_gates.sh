#!/usr/bin/env bash
# Train gates from scratch with the paper's recipe.
#
#   uv sync --extra train
#   demos/train_gates.sh gptoss  [lr]      # gpt-oss-20b, 8 GPUs   (paper: lr 0.1; sweep 0.01-0.1)
#   demos/train_gates.sh llama   [lr]      # Llama-3.3-70B, 8 GPUs (paper: lr 0.008)
#
# Both default to 8 GPUs, which with the configs' batch 1 x 8 accumulation steps per rank gives
# the paper's effective batch of 64 pairs per optimizer step. Fewer GPUs reproduce the same
# training only if the accumulation is raised to keep 64, e.g. for gpt-oss on one GPU:
#   NPROC=1 demos/train_gates.sh gptoss 0.1 gradient_accumulation_steps=64
#
# The training data is a private Hugging Face dataset (imperial-cpg/heads-of-security-dpo); set
# HF_TOKEN or `huggingface-cli login` first. The result lands in $OUT/gates/epoch_0.safetensors,
# which load_gated_model_local() / push_gates() take directly.
set -euo pipefail
FAMILY=${1:?usage: $0 gptoss|llama [lr]}
RUNS=${RUNS:-runs}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

case "${FAMILY}" in
  gptoss) CONFIG=configs/training/gpt_oss_20b.yaml;  LR=${2:-0.1};   NPROC=${NPROC:-8}; OUT=${RUNS}/gptoss20b_lr${LR} ;;
  llama)  CONFIG=configs/training/llama_3.3_70b.yaml; LR=${2:-0.008}; NPROC=${NPROC:-8}; OUT=${RUNS}/llama70b_lr${LR} ;;
  *) echo "unknown family ${FAMILY}"; exit 2 ;;
esac

exec uv run tune run --nproc_per_node "${NPROC}" src/heads_of_security/training/full_dpo_distributed.py \
    --config "${CONFIG}" output_dir="${OUT}" optimizer.lr="${LR}" "${@:3}"

#!/usr/bin/env bash
# Warm-start the 512 x 512 ten-class NFFA model from a weights-only checkpoint.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa_all10_centerresize_512}"
init_weights="${INIT_WEIGHTS:-${repo_root}/model/sem_nffa_all10_centerresize_512/ot/20260922_150017_sem_nffa_all10_centerresize_512_ot_amp_4gpu_gb12/model_20.pt}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size_train="${BATCH_SIZE_TRAIN:-2}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
num_epoch="${NUM_EPOCH:-29}"
run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_sem_nffa_all10_centerresize_512_ot_fp32_warmstart_model20_gb8}"
wandb_enabled="${WANDB_ENABLED:-True}"
wandb_project="${WANDB_PROJECT:-microscopic_inpainting}"
wandb_log_interval="${WANDB_LOG_INTERVAL:-10}"

if [[ ! -d "${dataset_root}/train" ]] || ! find -L "${dataset_root}/train" -type f -name '*.npy' -print -quit | grep -q .; then
    echo "Prepared center-resized SEM training data were not found at: ${dataset_root}" >&2
    exit 1
fi
if [[ ! -f "${init_weights}" ]]; then
    echo "Warm-start weights were not found at: ${init_weights}" >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "NPROC_PER_NODE=${nproc_per_node}; per-GPU batch=${batch_size_train}; accumulation=${gradient_accumulation_steps}; effective global batch=$((nproc_per_node * batch_size_train * gradient_accumulation_steps)); epochs=${num_epoch}"
echo "Dataset: ${dataset_root}"
echo "Warm-start weights: ${init_weights}; optimizer state will be fresh"
echo "W&B: enabled=${wandb_enabled}; project=${wandb_project}; log_interval=${wandb_log_interval}"
echo "AMP: disabled (FP32 training)"
echo "Run name: ${run_name}"

cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" torchrun --standalone \
    --nproc_per_node="${nproc_per_node}" \
    main_ddp.py \
    --no_cap \
    --init-weights "${init_weights}" \
    --run-name "${run_name}" \
    --opts \
    dataset sem_nffa_all10_centerresize_512 \
    model ot \
    train True \
    eval False \
    batch_size_train "${batch_size_train}" \
    num_epoch "${num_epoch}" \
    wandb "${wandb_enabled}" \
    wandb_project "${wandb_project}" \
    wandb_log_interval "${wandb_log_interval}" \
    amp False \
    gradient_accumulation_steps "${gradient_accumulation_steps}"

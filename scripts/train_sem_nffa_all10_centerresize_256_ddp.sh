#!/usr/bin/env bash
# Train the cleaned, center-cropped, resized 256 x 256 ten-class NFFA dataset.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa_all10_centerresize_256}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2}"
nproc_per_node="${NPROC_PER_NODE:-3}"
batch_size_train="${BATCH_SIZE_TRAIN:-4}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
num_epoch="${NUM_EPOCH:-50}"
run_name="${RUN_NAME:-}"
wandb_enabled="${WANDB_ENABLED:-True}"
wandb_project="${WANDB_PROJECT:-microscopic_inpainting}"
wandb_log_interval="${WANDB_LOG_INTERVAL:-10}"
amp_enabled="${AMP_ENABLED:-True}"

if [[ ! -d "${dataset_root}/train" ]] || ! find -L "${dataset_root}/train" -type f -name '*.npy' -print -quit | grep -q .; then
    echo "Prepared center-resized SEM training data were not found at: ${dataset_root}" >&2
    exit 1
fi

command=(
    torchrun --standalone --nproc_per_node="${nproc_per_node}" "${repo_root}/main_ddp.py"
    --no_cap
)
if [[ -n "${run_name}" ]]; then
    command+=(--run-name "${run_name}")
fi
command+=(
    --opts
    dataset sem_nffa_all10_centerresize_256
    model ot
    train True
    eval False
    batch_size_train "${batch_size_train}"
    num_epoch "${num_epoch}"
    wandb "${wandb_enabled}"
    wandb_project "${wandb_project}"
    wandb_log_interval "${wandb_log_interval}"
    amp "${amp_enabled}"
    gradient_accumulation_steps "${gradient_accumulation_steps}"
)

echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "NPROC_PER_NODE=${nproc_per_node}; per-GPU batch=${batch_size_train}; accumulation=${gradient_accumulation_steps}; effective global batch=$((nproc_per_node * batch_size_train * gradient_accumulation_steps)); epochs=${num_epoch}"
echo "Dataset: ${dataset_root}"
echo "W&B: enabled=${wandb_enabled}; project=${wandb_project}; log_interval=${wandb_log_interval}"
echo "AMP: enabled=${amp_enabled}"
if [[ -n "${run_name}" ]]; then
    echo "Run name: ${run_name}"
fi
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${command[@]}"

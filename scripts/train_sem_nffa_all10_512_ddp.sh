#!/usr/bin/env bash
# Launch distributed 512 x 512 OT flow-matching training on all ten NFFA-SEM categories.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 BATCH_SIZE_TRAIN=1 NUM_EPOCH=100 \
#     bash scripts/train_sem_nffa_all10_512_ddp.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa_all10_512}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size_train="${BATCH_SIZE_TRAIN:-1}"
num_epoch="${NUM_EPOCH:-100}"
no_cap="${NO_CAP:-1}"
run_name="${RUN_NAME:-}"
wandb_enabled="${WANDB_ENABLED:-True}"
wandb_project="${WANDB_PROJECT:-microscopic_inpainting}"
wandb_log_interval="${WANDB_LOG_INTERVAL:-10}"

if [[ ! -d "${dataset_root}/train" ]] || ! compgen -G "${dataset_root}/train/*.npy" > /dev/null; then
    echo "Prepared 512 px all-category SEM train patches were not found at: ${dataset_root}" >&2
    exit 1
fi

if [[ "${no_cap}" != "0" && "${no_cap}" != "1" ]]; then
    echo "NO_CAP must be 0 (smoke run) or 1 (full dataset)." >&2
    exit 1
fi

command=(
    torchrun
    --standalone
    --nproc_per_node="${nproc_per_node}"
    "${repo_root}/main_ddp.py"
)
if [[ "${no_cap}" == "1" ]]; then
    command+=(--no_cap)
fi
if [[ -n "${run_name}" ]]; then
    command+=(--run-name "${run_name}")
fi
command+=(
    --opts
    dataset sem_nffa_all10_512
    model ot
    train True
    eval False
    batch_size_train "${batch_size_train}"
    num_epoch "${num_epoch}"
    wandb "${wandb_enabled}"
    wandb_project "${wandb_project}"
    wandb_log_interval "${wandb_log_interval}"
)

echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "NPROC_PER_NODE=${nproc_per_node}; batch_size_train=${batch_size_train}; num_epoch=${num_epoch}; no_cap=${no_cap}"
echo "Dataset: ${dataset_root}"
echo "W&B: enabled=${wandb_enabled}; project=${wandb_project}; log_interval=${wandb_log_interval}"
if [[ -n "${run_name}" ]]; then
    echo "Run name: ${run_name}"
fi
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${command[@]}"

#!/usr/bin/env bash
# Launch full-dataset distributed OT flow-matching training on prepared SEM patches.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 BATCH_SIZE_TRAIN=16 NUM_EPOCH=20 \
#     bash scripts/train_sem_nffa_ddp.sh
#
# Set NO_CAP=0 for the short 21-batch smoke run used to validate a new setup.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size_train="${BATCH_SIZE_TRAIN:-16}"
num_epoch="${NUM_EPOCH:-20}"
no_cap="${NO_CAP:-1}"

if [[ ! -d "${dataset_root}/train" ]] || ! compgen -G "${dataset_root}/train/*.npy" > /dev/null; then
    echo "Prepared SEM train patches were not found at: ${dataset_root}" >&2
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
command+=(
    --opts
    dataset sem_nffa
    model ot
    train True
    eval False
    batch_size_train "${batch_size_train}"
    num_epoch "${num_epoch}"
)

echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "NPROC_PER_NODE=${nproc_per_node}; batch_size_train=${batch_size_train}; num_epoch=${num_epoch}; no_cap=${no_cap}"
echo "Dataset: ${dataset_root}"
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${command[@]}"

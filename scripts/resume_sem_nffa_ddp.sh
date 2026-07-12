#!/usr/bin/env bash
# Resume a SEM DDP run from the latest epoch-complete training-state checkpoint.
#
# Example: continue a run from completed epoch 19 through epoch 99 (100 total):
#   RUN_NAME=20260712_150000_sem_nffa_ot TARGET_EPOCHS=100 \
#     CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
#     bash scripts/resume_sem_nffa_ddp.sh
#
# The target is a total epoch count, not an additional epoch count. This launcher
# requires training_state_latest.pt, which is written once per completed epoch by
# the updated flow-matching trainer.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size_train="${BATCH_SIZE_TRAIN:-16}"
target_epochs="${TARGET_EPOCHS:-100}"
run_name="${RUN_NAME:-}"

if [[ ! -d "${dataset_root}/train" ]] || ! compgen -G "${dataset_root}/train/*.npy" > /dev/null; then
    echo "Prepared SEM train patches were not found at: ${dataset_root}" >&2
    exit 1
fi

if [[ -z "${run_name}" ]]; then
    latest_run_file="${repo_root}/model/sem_nffa/ot/latest_run.txt"
    if [[ ! -f "${latest_run_file}" ]]; then
        echo "Set RUN_NAME to the run directory you want to resume." >&2
        exit 1
    fi
    run_name="$(<"${latest_run_file}")"
fi

resume_checkpoint="${RESUME_CHECKPOINT:-${repo_root}/model/sem_nffa/ot/${run_name}/training_state_latest.pt}"
if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "Resumable training state not found: ${resume_checkpoint}" >&2
    echo "Legacy model_*.pt and model_final.pt files do not include optimizer state." >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "NPROC_PER_NODE=${nproc_per_node}; batch_size_train=${batch_size_train}; target_epochs=${target_epochs}"
echo "Run name: ${run_name}"
echo "Resume checkpoint: ${resume_checkpoint}"
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" torchrun --standalone \
    --nproc_per_node="${nproc_per_node}" \
    main_ddp.py \
    --no_cap \
    --resume-checkpoint "${resume_checkpoint}" \
    --run-name "${run_name}" \
    --opts \
    dataset sem_nffa \
    model ot \
    train True \
    eval False \
    batch_size_train "${batch_size_train}" \
    num_epoch "${target_epochs}"

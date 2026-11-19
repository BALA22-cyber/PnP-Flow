#!/usr/bin/env bash
# Resume a cleaned center-resized 512 x 512 NFFA DDP run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa_all10_centerresize_512}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2}"
nproc_per_node="${NPROC_PER_NODE:-3}"
batch_size_train="${BATCH_SIZE_TRAIN:-2}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"
target_epochs="${TARGET_EPOCHS:-50}"
run_name="${RUN_NAME:-}"
wandb_enabled="${WANDB_ENABLED:-True}"
wandb_project="${WANDB_PROJECT:-microscopic_inpainting}"
wandb_log_interval="${WANDB_LOG_INTERVAL:-10}"
amp_enabled="${AMP_ENABLED:-True}"

if [[ ! -d "${dataset_root}/train" ]] || ! find -L "${dataset_root}/train" -type f -name '*.npy' -print -quit | grep -q .; then
    echo "Prepared center-resized SEM training data were not found at: ${dataset_root}" >&2
    exit 1
fi
if [[ -z "${run_name}" ]]; then
    latest_run_file="${repo_root}/model/sem_nffa_all10_centerresize_512/ot/latest_run.txt"
    if [[ ! -f "${latest_run_file}" ]]; then
        echo "Set RUN_NAME to the run directory to resume." >&2
        exit 1
    fi
    run_name="$(<"${latest_run_file}")"
fi

resume_checkpoint="${RESUME_CHECKPOINT:-${repo_root}/model/sem_nffa_all10_centerresize_512/ot/${run_name}/training_state_latest.pt}"
if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "Resumable training state not found: ${resume_checkpoint}" >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "NPROC_PER_NODE=${nproc_per_node}; per-GPU microbatch=${batch_size_train}; accumulation=${gradient_accumulation_steps}; effective global batch=$((nproc_per_node * batch_size_train * gradient_accumulation_steps)); target epochs=${target_epochs}"
echo "Run name: ${run_name}"
echo "Resume checkpoint: ${resume_checkpoint}"
echo "W&B: enabled=${wandb_enabled}; project=${wandb_project}; log_interval=${wandb_log_interval}"
echo "AMP: enabled=${amp_enabled}"
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" torchrun --standalone \
    --nproc_per_node="${nproc_per_node}" \
    main_ddp.py \
    --no_cap \
    --resume-checkpoint "${resume_checkpoint}" \
    --run-name "${run_name}" \
    --opts \
    dataset sem_nffa_all10_centerresize_512 \
    model ot \
    train True \
    eval False \
    batch_size_train "${batch_size_train}" \
    num_epoch "${target_epochs}" \
    wandb "${wandb_enabled}" \
    wandb_project "${wandb_project}" \
    wandb_log_interval "${wandb_log_interval}" \
    amp "${amp_enabled}" \
    gradient_accumulation_steps "${gradient_accumulation_steps}"

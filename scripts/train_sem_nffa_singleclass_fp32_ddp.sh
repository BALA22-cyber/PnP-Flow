#!/usr/bin/env bash
# FP32 NFFA class-specific fine-tuning from the shared model_20 initialization.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa_all10_centerresize_512}"
nffa_class="${NFFA_CLASS:-Patterned_surface}"
init_weights="${INIT_WEIGHTS:-${repo_root}/model/sem_nffa_all10_centerresize_512/ot/20260922_150017_sem_nffa_all10_centerresize_512_ot_amp_4gpu_gb12/model_20.pt}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size_train="${BATCH_SIZE_TRAIN:-1}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"
num_epoch="${NUM_EPOCH:-130}"
run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_sem_nffa_${nffa_class}_512_fp32_warmstart_model20_gb8_130ep}"
wandb_enabled="${WANDB_ENABLED:-True}"
wandb_project="${WANDB_PROJECT:-microscopic_inpainting}"
wandb_log_interval="${WANDB_LOG_INTERVAL:-10}"

for split in train val test; do
    split_dir="${dataset_root}/${split}/${nffa_class}"
    if [[ ! -d "${split_dir}" ]] || ! find -L "${split_dir}" -type f -name '*.npy' -print -quit | grep -q .; then
        echo "Prepared NFFA class '${nffa_class}' not found for split '${split}': ${split_dir}" >&2
        exit 1
    fi
done
if [[ ! -f "${init_weights}" ]]; then
    echo "Warm-start weights were not found at: ${init_weights}" >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "NFFA class: ${nffa_class}"
echo "NPROC_PER_NODE=${nproc_per_node}; per-GPU microbatch=${batch_size_train}; accumulation=${gradient_accumulation_steps}; effective global batch=$((nproc_per_node * batch_size_train * gradient_accumulation_steps)); epochs=${num_epoch}"
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
    nffa_class "${nffa_class}" \
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

#!/usr/bin/env bash
# Resume a 512 x 512 NFFA single-class FP32 DDP run without changing its class.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nffa_class="${NFFA_CLASS:?Set NFFA_CLASS to the class used by the checkpoint}"
run_name="${RUN_NAME:?Set RUN_NAME to the existing class run directory}"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa_all10_centerresize_512}"
resume_checkpoint="${RESUME_CHECKPOINT:-${repo_root}/model/sem_nffa_all10_centerresize_512/ot/${run_name}/training_state_latest.pt}"
target_epochs="${TARGET_EPOCHS:-130}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size_train="${BATCH_SIZE_TRAIN:-1}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"
wandb_enabled="${WANDB_ENABLED:-True}"
wandb_project="${WANDB_PROJECT:-microscopic_inpainting}"
wandb_log_interval="${WANDB_LOG_INTERVAL:-10}"
torchlearn_bin="${TORCHLEARN_BIN:-/data2/marci/bselvaku/conda/envs/torchlearn/bin}"

for split in train val test; do
    if [[ ! -d "${dataset_root}/${split}/${nffa_class}" ]]; then
        echo "Missing ${split} data for class '${nffa_class}'" >&2
        exit 1
    fi
done
if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "Resumable class checkpoint not found: ${resume_checkpoint}" >&2
    exit 1
fi

checkpoint_info="$("${torchlearn_bin}/python" - "${resume_checkpoint}" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(f"{state.get('nffa_class', '')} {state.get('completed_epoch', -1)}")
PY
)"
read -r checkpoint_class completed_epoch <<< "${checkpoint_info}"
if [[ "${checkpoint_class}" != "${nffa_class}" ]]; then
    echo "Checkpoint class '${checkpoint_class}' does not match requested class '${nffa_class}'." >&2
    exit 1
fi
if (( completed_epoch + 1 >= target_epochs )); then
    echo "Checkpoint already completed epoch ${completed_epoch}; target is ${target_epochs} epochs." >&2
    exit 1
fi

echo "Resuming ${nffa_class} run ${run_name} from epoch $((completed_epoch + 1)) to $((target_epochs - 1))"
echo "GPUs=${cuda_visible_devices}; processes=${nproc_per_node}; microbatch=${batch_size_train}; accumulation=${gradient_accumulation_steps}; AMP=False"
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${torchlearn_bin}/torchrun" --standalone \
    --nproc_per_node="${nproc_per_node}" \
    main_ddp.py \
    --no_cap \
    --resume-checkpoint "${resume_checkpoint}" \
    --run-name "${run_name}" \
    --opts \
    dataset sem_nffa_all10_centerresize_512 \
    nffa_class "${nffa_class}" \
    model ot \
    train True \
    eval False \
    batch_size_train "${batch_size_train}" \
    num_epoch "${target_epochs}" \
    wandb "${wandb_enabled}" \
    wandb_project "${wandb_project}" \
    wandb_log_interval "${wandb_log_interval}" \
    amp False \
    gradient_accumulation_steps "${gradient_accumulation_steps}"

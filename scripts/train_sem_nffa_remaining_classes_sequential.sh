#!/usr/bin/env bash
# Train the nine other NFFA classes serially with the Patterned_surface FP32 setup.
# Pass class names to run only selected classes; --dry-run prints the plan.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${DATASET_ROOT:-${repo_root}/data/sem_nffa_all10_centerresize_512}"
run_group="${RUN_GROUP:-nffa512_singleclass_model20_fp32_gb8_130ep}"
target_epochs="${TARGET_EPOCHS:-130}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size_train="${BATCH_SIZE_TRAIN:-1}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"
dry_run="${DRY_RUN:-0}"

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if (( $# > 0 )); then
    classes=("$@")
else
    # Largest classes first. Patterned_surface already has its own run.
    classes=(
        MEMS_devices_and_electrodes Particles Nanowires Tips Biological
        Powder Films_Coated_Surface Porous_Sponge Fibres
    )
fi

export PATH="/data2/marci/bselvaku/conda/envs/torchlearn/bin:${PATH}"
mkdir -p "${repo_root}/logs"
for nffa_class in "${classes[@]}"; do
    case "${nffa_class}" in
        Biological|Fibres|Films_Coated_Surface|MEMS_devices_and_electrodes|Nanowires|Particles|Porous_Sponge|Powder|Tips) ;;
        *) echo "Unsupported remaining NFFA class: ${nffa_class}" >&2; exit 1 ;;
    esac
    for split in train val test; do
        if [[ ! -d "${dataset_root}/${split}/${nffa_class}" ]]; then
            echo "Missing ${split} data for class '${nffa_class}'" >&2
            exit 1
        fi
    done

    train_count="$(find -L "${dataset_root}/train/${nffa_class}" -maxdepth 1 -type f -name '*.npy' | wc -l)"
    per_rank_batches=$((train_count / nproc_per_node / batch_size_train))
    if (( per_rank_batches < 1 )); then
        echo "Class '${nffa_class}' has too few images for this DDP batch configuration." >&2
        exit 1
    fi
    updates_per_epoch=$(((per_rank_batches + gradient_accumulation_steps - 1) / gradient_accumulation_steps))
    run_name="${run_group}_${nffa_class}"
    model_dir="${repo_root}/model/sem_nffa_all10_centerresize_512/ot/${run_name}"
    log_path="${repo_root}/logs/${run_name}.log"

    if [[ -f "${model_dir}/model_final.pt" ]]; then
        action=skip
    elif [[ -f "${model_dir}/training_state_latest.pt" ]]; then
        action=resume
    else
        action=start
    fi
    echo "${nffa_class}: ${train_count} train images, ${updates_per_epoch} optimizer updates/epoch, approximately $((updates_per_epoch * target_epochs)) updates total; action=${action}; run=${run_name}"
    if [[ "${dry_run}" == 1 || "${action}" == skip ]]; then
        continue
    fi

    if [[ "${action}" == resume ]]; then
        env NFFA_CLASS="${nffa_class}" RUN_NAME="${run_name}" TARGET_EPOCHS="${target_epochs}" \
            CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" NPROC_PER_NODE="${nproc_per_node}" \
            BATCH_SIZE_TRAIN="${batch_size_train}" GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}" \
            bash "${repo_root}/scripts/resume_sem_nffa_singleclass_fp32_ddp.sh" 2>&1 | tee -a "${log_path}"
    else
        env NFFA_CLASS="${nffa_class}" RUN_NAME="${run_name}" NUM_EPOCH="${target_epochs}" \
            CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" NPROC_PER_NODE="${nproc_per_node}" \
            BATCH_SIZE_TRAIN="${batch_size_train}" GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}" \
            bash "${repo_root}/scripts/train_sem_nffa_singleclass_fp32_ddp.sh" 2>&1 | tee -a "${log_path}"
    fi
done

#!/usr/bin/env bash
# Test the trained 128 x 128 all-category SEM prior on the original authors'
# random-inpainting problem. main.py uses a fixed random mask with 70% missing
# pixels and Gaussian sigma=0.01 for this problem.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="${RUN_NAME:-20260713_112747_sem_nffa_all10_ot}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
max_batch="${MAX_BATCH:-1}"
batch_size_ip="${BATCH_SIZE_IP:-4}"

checkpoint="${repo_root}/model/sem_nffa_all10/ot/${run_name}/model_final.pt"
if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 1
fi

echo "Checkpoint: ${checkpoint}"
echo "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "Random inpainting: max_batch=${max_batch}; batch_size_ip=${batch_size_ip}"

cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" python main.py --opts \
    dataset sem_nffa_all10 \
    model ot \
    model_run "${run_name}" \
    train False \
    eval True \
    eval_split test \
    problem random_inpainting \
    method pnp_flow \
    lr_pnp 1.0 \
    alpha 0.01 \
    num_samples 5 \
    max_batch "${max_batch}" \
    batch_size_ip "${batch_size_ip}" \
    steps_pnp 100 \
    save_results True

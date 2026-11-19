#!/usr/bin/env bash
# Run the paired inpainting evaluation after both four-GPU training jobs release the GPUs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
all_class_run="20260923_140640_sem_nffa_all10_centerresize_512_ot_fp32_warmstart_model20_gb8_29ep"
single_class_run="20260923_203800_sem_nffa_Patterned_surface_512_fp32_from_model20_gb8_130ep"

echo "Waiting for all-class and Patterned_surface training sessions to finish."
while tmux has-session -t nffa_center512_fp32_warmstart 2>/dev/null \
   || tmux has-session -t nffa_patterned_fp32 2>/dev/null; do
    sleep 60
done

export PATH="/data2/marci/bselvaku/conda/envs/torchlearn/bin:/usr/bin:/bin"
cd "${repo_root}"
python scripts/evaluate_sem_nffa_singleclass_inpainting.py \
    --repo-root "${repo_root}" \
    --class-name Patterned_surface \
    --all-class-run "${all_class_run}" \
    --single-class-run "${single_class_run}" \
    --test-index 0 \
    --mask-sizes 64,128,256 \
    --steps 100 \
    --noise-std 0.001 \
    --lr-pnp 1.0 \
    --alpha 1.0 \
    --num-samples 1

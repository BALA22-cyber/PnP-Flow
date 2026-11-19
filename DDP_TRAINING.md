# Multi-GPU Training With `main_ddp.py`

This project now has a separate DistributedDataParallel training entrypoint:

```text
main_ddp.py
```

Use `main.py` for the existing single-GPU training/evaluation flow. Use `main_ddp.py` only for multi-GPU training of the OT flow-matching model.

## Environment Setup

On the target machine:

```bash
conda create -n torchlearn python=3.11 -y
conda activate torchlearn
cd /path/to/microscopic_inpainting/modified_code/PnP-Flow
```

Install a CUDA-enabled PyTorch build appropriate for the machine, then install this package:

```bash
pip install -r requirements.txt
pip install -e .
```

Verify PyTorch sees the GPUs:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

For a 4x V100 machine, this should show `torch.cuda.device_count()` as `4`.

## Launch Command

Run from the `modified_code/PnP-Flow` directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 main_ddp.py \
  --no_cap \
  --opts dataset celeba model ot train True eval False batch_size_train 16 num_epoch 200
```

`--nproc_per_node=4` means one training process per GPU.

## Batch Size Meaning

In DDP, `batch_size_train` is the per-GPU batch size.

```text
batch_size_train 16 on 4 GPUs = effective global batch size 64
batch_size_train 32 on 4 GPUs = effective global batch size 128
```

Start with:

```text
batch_size_train 16
```

If memory is stable on the V100s, try:

```text
batch_size_train 32
```

## Capped vs Full Dataset Training

Without `--no_cap`, training stops after 21 batches per epoch.

With `--no_cap`, training uses the full DataLoader.

For CelebA with `batch_size_train 16` on 4 GPUs:

```text
global batch size = 64
~162,769 training images / 64 = ~2,544 optimizer steps per full epoch
```

## Outputs

Only rank 0 writes logs, samples, and checkpoints. Output layout remains the same as the current timestamped run format:

```text
results/<dataset>/<model>/<timestamp_dataset_model>/
model/<dataset>/<model>/<timestamp_dataset_model>/
```

Example:

```text
results/celeba/ot/20260621_153000_celeba_ot/
model/celeba/ot/20260621_153000_celeba_ot/model_final.pt
```

The `latest_run.txt` pointer is updated only after `model_final.pt` is successfully written.

## Evaluation

Use the normal single-GPU/eval script after training:

```bash
python main.py --opts dataset celeba model ot train False eval True \
  model_run <timestamp_dataset_model> \
  problem inpainting method pnp_flow eval_split test
```

If `model_run` is omitted, `main.py` uses:

```text
model/<dataset>/<model>/latest_run.txt
```

## Weights & Biases Logging

The SEM DDP training and resume scripts enable W&B logging by default with:

```text
project: microscopic_inpainting
entity: account associated with the logged-in API key
batch log interval: 10 optimizer steps
```

Logged metrics include:

- DDP-averaged batch training loss
- optimizer step, epoch, iteration, and learning rate
- DDP-averaged epoch loss
- epoch duration and global samples processed
- global samples per second
- peak allocated GPU memory on rank 0
- W&B automatic host, CPU, GPU, and process metrics

Only rank 0 creates and writes the W&B run. Other DDP ranks participate in
metric reduction but do not create duplicate runs.

The launchers can be configured through environment variables:

```bash
WANDB_ENABLED=True \
WANDB_PROJECT=microscopic_inpainting \
WANDB_LOG_INTERVAL=10 \
bash scripts/train_sem_nffa_all10_512_ddp.sh
```

Disable W&B for a particular run with:

```bash
WANDB_ENABLED=False bash scripts/train_sem_nffa_all10_512_ddp.sh
```

For a direct `main.py` or `main_ddp.py` command, add these options:

```text
wandb True wandb_project microscopic_inpainting wandb_log_interval 10
```

Each result directory stores `wandb_run_id.txt`. Resuming a run reuses this ID
and continues the same dashboard run. An older run without that file creates a
new W&B dashboard run on its first W&B-enabled resume; subsequent resumes reuse
the new ID.

## Center-resized SEM datasets

`scripts/prepare_nffa_clean_resized.py` builds the 512×512 dataset from raw
images and an existing source split manifest. It removes detected footer bands,
center-crops to a square, resizes, and normalizes each image. Use `--help` for
the required input, output, manifest, and contact-sheet paths; pass `--limit 0`
to process the full manifest.

`scripts/prepare_nffa_centerresize_256.py` builds the 256×256 dataset from the
existing cleaned PNGs and processing manifest, preserving their source splits.
Its default input path is specific to the original machine; use `--clean-root`,
`--source-manifest`, and `--output-root` when using another filesystem. Both
datasets store normalized grayscale float16 NPY files and must be available
under `data/sem_nffa_all10_centerresize_256` or `data/sem_nffa_all10_centerresize_512`.

Launch training with `bash scripts/train_sem_nffa_all10_centerresize_256_ddp.sh`
or `bash scripts/train_sem_nffa_all10_centerresize_512_ddp.sh`. The launchers
accept `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `BATCH_SIZE_TRAIN`,
`GRADIENT_ACCUMULATION_STEPS`, `NUM_EPOCH`, `AMP_ENABLED`, and W&B environment
variables. Select a per-GPU batch size that fits available memory. `DATASET_ROOT`
checks for the dataset but does not change the loader path; put or link the
dataset under `data/` as described above.

The `sem_nffa_centerresize_256_box_inpainting.ipynb` and
`sem_nffa_centerresize_512_box_inpainting.ipynb` notebooks demonstrate box
inpainting on test images. The 256 notebook uses the latest completed run by
default; set `RUN_NAME` and optionally `CHECKPOINT_NAME` to inspect an
intermediate checkpoint. The 512 notebook has a specific run and checkpoint
configured in its setup cell. Checkpoints and datasets are stored separately
from Git and their local paths may need updating on another machine.

## General notes

- `main_ddp.py` currently supports OT flow-matching training only: `model ot`.
- Keep using `main.py` for inference/evaluation.
- Do not set `CUDA_VISIBLE_DEVICES` inside the script. Set it in the shell command.
- If NCCL fails, first confirm all GPUs are visible with `nvidia-smi` and the PyTorch GPU check above.

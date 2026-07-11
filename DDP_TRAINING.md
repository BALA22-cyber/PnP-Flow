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

## Notes

- `main_ddp.py` currently supports OT flow-matching training only: `model ot`.
- Keep using `main.py` for inference/evaluation.
- Do not set `CUDA_VISIBLE_DEVICES` inside the script. Set it in the shell command.
- If NCCL fails, first confirm all GPUs are visible with `nvidia-smi` and the PyTorch GPU check above.

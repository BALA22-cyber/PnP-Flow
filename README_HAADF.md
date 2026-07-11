# HAADF / Microscopy Inpainting Additions

This modified copy of `annegnx/PnP-Flow` adds a small HAADF pathway for testing PnP-Flow Matching on grayscale microscopy patches.

## What was added

- `config/dataset_config/haadf2.yaml` with `dim_image: 128`, `num_channels: 1`.
- `.npy` patch dataset support in `pnpflow/dataloaders.py`.
- `FixedMaskInpainting` in `pnpflow/degradations.py`.
- Synthetic fixed HAADF inpainting problems in `main.py`:
  - `haadf_blob_inpainting`
  - `haadf_line_inpainting`
  - `haadf_random_inpainting`
- Data-consistency copy-back inside `pnpflow/methods/pnp_flow.py` for any problem containing `inpainting`.
- FID bypass for custom HAADF training in `pnpflow/train_flow_matching.py`.
- Grayscale-safe visualization/postprocessing and missing-pixel PSNR in `pnpflow/utils.py`.
- `scripts/prepare_haadf_patches.py` to convert an HDF5 HAADF image into PnP-Flow-ready `.npy` patches.

## Expected dataset layout

```text
data/haadf2/train/*.npy
data/haadf2/val/*.npy
data/haadf2/test/*.npy
```

Each `.npy` file should be a normalized grayscale patch in `[-1, 1]`, shape `128 x 128` or `1 x 128 x 128`.

## Prepare patches from HAADF_21.h5

From the repo root:

```bash
python scripts/prepare_haadf_patches.py \
  --h5 data/HAADF_21.h5 \
  --out data/haadf2 \
  --patch-size 128 \
  --stride 64
```

The default HDF5 dataset path is:

```text
Measurement_000/Channel_000/HAADF/HAADF
```

## Tiny training sanity check

```bash
python main.py --opts \
  dataset haadf2 \
  model ot \
  train True \
  eval False \
  batch_size_train 4 \
  num_epoch 1 \
  lr 0.0001
```

This should create:

```text
model/haadf2/ot/model_final.pt
```

## Longer but still small HAADF prior training

```bash
python main.py --opts \
  dataset haadf2 \
  model ot \
  train True \
  eval False \
  batch_size_train 8 \
  num_epoch 20 \
  lr 0.0001
```

This is a first test only. With one/few microscopy images, do not treat the result as a broad learned microscopy prior.

## Run synthetic HAADF inpainting

Blob mask:

```bash
python main.py --opts \
  dataset haadf2 \
  model ot \
  train False \
  eval True \
  eval_split test \
  problem haadf_blob_inpainting \
  method pnp_flow \
  batch_size_ip 4 \
  max_batch 1 \
  save_results True
```

Line mask:

```bash
python main.py --opts \
  dataset haadf2 \
  model ot \
  train False \
  eval True \
  eval_split test \
  problem haadf_line_inpainting \
  method pnp_flow \
  batch_size_ip 4 \
  max_batch 1 \
  save_results True
```

Random fixed mask:

```bash
python main.py --opts \
  dataset haadf2 \
  model ot \
  train False \
  eval True \
  eval_split test \
  problem haadf_random_inpainting \
  method pnp_flow \
  batch_size_ip 4 \
  max_batch 1 \
  save_results True
```

## Outputs to check

Results are saved under:

```text
results/haadf2/ot/<problem>/pnp_flow/test/<method-config-folder>/
```

Useful files include:

- final clean / noisy / restored images
- `psnr_rec_*`
- `ssim_rec_*`
- `masked_psnr_rec_*`

For inpainting, `masked_psnr_rec_average.txt` is more meaningful than full-image PSNR because known pixels are copied back exactly.

## Important scientific caution

This patch is for a controlled benchmark. Compare PnP-Flow against the microscopy-aware baselines on identical synthetic masks before using it on real corrupted `graphene_2.h5` regions. For real corruption, build a real mask first, dilate/clean it, normalize from clean lattice regions, and always preserve measured pixels.

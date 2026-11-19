# Inpainting workflows and interpretation

## Training and inference

OT flow-matching training learns a clean-image prior from noise and clean images. Masks are applied later when PnP-Flow solves the inpainting problem. Training loss measures velocity-prediction error; generated sample figures provide a prior sanity check and are not inpainting reconstructions.

For a fixed mask, `FixedMaskInpainting` uses 1 for observed pixels and 0 for missing pixels. The inpainting solver copies the observed values back after each update. When observations include added noise, this preserves those noisy observations. Masked PSNR measures error within missing pixels; full-image scores can look high simply because most observed pixels were retained.

## Patch, tile, and resized-image workflows

- The center-resized 256 and 512 box notebooks load a prepared held-out test array, mask a square, and reconstruct it at the model's input size.
- `test_sem_nffa_model.ipynb` includes a held-out 128×128 patch experiment. The full-frame SEM square and random notebooks use overlapping native-resolution 128×128 tiles, then combine tile predictions.
- `nffa_full_raw_resized_128_inpainting.ipynb` instead resizes the complete raw image and mask to 128×128 for inference, then enlarges the result for display. This is a separate experiment from native-resolution tiling and does not match the random-crop training preprocessing.
- The HAADF square and random full-frame notebooks apply a SEM prior to HAADF data using native-resolution tiles. The resized HAADF notebook uses a whole-frame resize. These are cross-domain experiments, so quality on one HAADF frame does not establish general microscopy performance.
- The Patterned Surface and MEMS notebooks evaluate class-specific 512 models; check their setup cells for the selected model, dataset split, mask, and preprocessing.

Full-frame results can depend on tile overlap and blending, mask size, normalization, random seed, and solver settings. A clean test image with a synthetic mask provides a reference for measuring the missing region. For genuinely damaged images, the original missing content is unknown and a suitable external mask is needed. Inspect notebook setup cells and saved metrics for the precise experiment; saved plots alone do not identify every setting.

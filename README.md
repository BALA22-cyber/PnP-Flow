# PnP-Flow
![build](https://github.com/annegnx/PnP-Flow/actions/workflows/main.yml/badge.svg)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)


This GitHub repository contains the code for our ICLR 2025 PnP-Flow [paper](https://arxiv.org/pdf/2410.02423), a method combining PnP methods with Flow Matching pretrained models for solving image restoration problems. **Try out the [demo](https://github.com/annegnx/PnP-Flow/blob/main/demo/demo.ipynb)!**

<img src="demo/algo.png" style="width: 50%; display: flex; justify-content: center;"/>
<img src="demo/pnp_flow.png" scale=0.8/>

## Microscopy training and notebooks

This fork adds NFFA-SEM and HAADF data preparation, distributed training, and
inpainting notebooks. The datasets and trained model checkpoints are stored
separately from Git; the notebook figures and saved outputs in this repository
can be viewed without rerunning inference.

For the environment, install a CUDA-enabled PyTorch build appropriate to your
machine, then run `pip install -r requirements.txt` and `pip install -e .` from
the repository root. Use a Jupyter kernel in that environment to run notebooks.
See [DDP_TRAINING.md](DDP_TRAINING.md) for the full multi-GPU setup and
[README_HAADF.md](README_HAADF.md) for the HAADF workflow.

### Prepare SEM data

The center-resized datasets contain one grayscale, normalized image per source,
with the existing source-level train, validation, and test splits. Put the NPY
files under `data/sem_nffa_all10_centerresize_256/{train,val,test}/<category>/`
or the corresponding `sem_nffa_all10_centerresize_512` directory.

- `scripts/prepare_nffa_clean_resized.py` prepares the 512×512 data from raw
  images and a source split manifest. Run `python scripts/prepare_nffa_clean_resized.py --help`
  for required paths and options; use `--limit 0` for the complete manifest.
- `scripts/prepare_nffa_centerresize_256.py` prepares 256×256 data from cleaned
  images and their processing manifest. Run `python scripts/prepare_nffa_centerresize_256.py --help`
  to specify input and output locations; its default input path is specific to
  the original machine.

### Train a center-resized model

Run from the repository root, adjusting the GPU list and batch size for your
machine. Each DDP process uses one GPU, and the effective batch size is the
per-GPU batch size multiplied by the number of GPUs and accumulation steps.

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BATCH_SIZE_TRAIN=2 \
  WANDB_ENABLED=False bash scripts/train_sem_nffa_all10_centerresize_256_ddp.sh
```

Use `scripts/train_sem_nffa_all10_centerresize_512_ddp.sh` for the 512×512
dataset. Both launchers support `NUM_EPOCH`, `GRADIENT_ACCUMULATION_STEPS`,
`AMP_ENABLED`, and W&B options; their defaults are listed in each script.
Training saves checkpoints under `model/<dataset>/ot/<run>/` and results under
`results/<dataset>/ot/<run>/`. These directories are excluded from Git.

### Run box-inpainting notebooks

Open the notebook from the repository root or its `notebooks/` directory, select
the environment's Python kernel, and run the cells in order:

- [256×256 box inpainting](notebooks/sem_nffa_centerresize_256_box_inpainting.ipynb)
  loads a held-out test image and selects the latest completed training run by
  default. Set `RUN_NAME` and `CHECKPOINT_NAME` in the setup cell to use a
  specific run or intermediate checkpoint.
- [512×512 box inpainting](notebooks/sem_nffa_centerresize_512_box_inpainting.ipynb)
  uses the same type of test and has a specific run and checkpoint configured
  in its setup cell; change those values if your checkpoint is elsewhere.

Set `TEST_SAMPLE_INDEX` and `BOX_SIZE` in the setup cell to choose a test patch
and mask size. The notebooks display the input and reconstruction, report PSNR
within the missing region, and save figures, arrays, and metrics under
`notebooks/outputs/`. Running them requires the matching dataset and model
checkpoint; existing saved notebook outputs can be inspected without them.

### Dataset versions and splits

The SEM datasets split source images approximately 80%/10%/10% into train,
validation, and test sets, so crops from one source stay in the same split.
Class folders represent the ten NFFA categories. `sem_nffa` uses three classes:
Films/Coated Surface, Nanowires, and Particles. Each dataset's manifest, when
available, records the exact source assignment; the original manifest is needed
to reproduce the exact splits. The center-resize scripts consume an existing
manifest rather than creating one.

| Dataset | Preparation | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `sem_nffa` | Up to four native 128×128 random crops per source; three classes | 25,820 | 3,220 | 3,236 |
| `sem_nffa_all10` | Up to four native 128×128 random crops; ten classes | 68,044 | 8,496 | 8,530 |
| `sem_nffa_all10_512` | Four native 512×512 random crops; ten classes | 68,040 | 8,496 | 8,532 |
| `sem_nffa_all10_centerresize_512` | One cleaned, center-cropped, resized image per source | 17,011 | 2,123 | 2,133 |
| `sem_nffa_all10_centerresize_256` | Same cleaned sources and splits, resized to 256×256 | 17,011 | 2,123 | 2,133 |

Counts describe the prepared datasets used in these experiments; a new source
manifest or skipped images can change them. The center-resized datasets exclude
one validation image whose percentile range was constant. The random-crop
datasets can be prepared with `scripts/prepare_sem_patches.py`; use `--help`
to choose patch size, number of crops, split fractions, and seed. HAADF patch
preparation and its spatial split are described in [README_HAADF.md](README_HAADF.md).

### Other training and evaluation scripts

The `train_sem_nffa_ddp.sh`, `train_sem_nffa_all10_ddp.sh`, and
`train_sem_nffa_all10_512_ddp.sh` launchers target the corresponding random-crop
datasets; their `resume_...` counterparts resume from a training-state
checkpoint. The center-resized 512 launcher also has a resume script and an FP32
warm-start script. `train_sem_nffa_singleclass_fp32_ddp.sh` fine-tunes the
center-resized 512 model for one class, using `NFFA_CLASS` and `INIT_WEIGHTS`;
`resume_sem_nffa_singleclass_fp32_ddp.sh` resumes that run. Inspect the scripts'
defaults and set their environment variables for your GPUs and checkpoint paths.
The sequential class launcher and queued evaluation script contain experiment-
specific assumptions and paths that should be reviewed before use.

For class comparisons, `python scripts/evaluate_sem_nffa_singleclass_inpainting.py --help`
lists the required all-class and single-class run names and the mask options.
All these workflows require the corresponding data and checkpoints to be
present locally.

### Other notebooks and portability

- `test_sem_nffa_model.ipynb` evaluates the three-class 128 model on test
  patches; `nffa_model_square_mask.ipynb` applies a square mask to a raw SEM image.
- `nffa_full_raw_random_inpainting.ipynb` and
  `nffa_full_raw_resized_128_inpainting.ipynb` examine full raw SEM images;
  `sem_nffa_resized_128_multisize_square_inpainting.ipynb` tests several box sizes.
- `test_haadf_inpainting.ipynb`, `haadf_graphene_full_raw_random_inpainting.ipynb`,
  and `haadf_resized_128_multisize_square_inpainting.ipynb` use HAADF data with
  a SEM prior. `README_HAADF.md` describes the input format.
- `sem_nffa_patterned_surface_512_inpainting.ipynb`, its random and rotation
  variants, and `sem_nffa_mems_devices_512_inpainting.ipynb` evaluate
  class-specific 512 models. `celeba_pretrained_vs_local_epoch15_multisize_inpainting.ipynb`
  compares CelebA checkpoints.

These notebooks may hardcode paths, run names, checkpoints, GPU requirements,
or image choices from the original machine. Review their setup cells before
running. Install Jupyter and an `ipykernel` in the chosen environment if they
are not already available. The example counts and saved outputs document prior
experiments; they do not supply the original raw data, split manifests, or model
weights. Reproducing a result requires those matching artifacts and compatible
software and hardware.

For more detail, see [dataset preparation and splits](docs/datasets.md),
[inpainting workflows and metrics](docs/inpainting_workflows.md), and the
[microscopy dataset comparison](docs/dataset_comparison.md).

## 1. Getting started
To get started, clone the repository and install ```pnpflow``` via pip

```
cd PnP-Flow
pip install -e .
```
### 1.1. Requirements

- torch 1.13.1 (or later)
- torchvision
- tqdm
- numpy
- pandas
- pyyaml
- scipy
- torchdiffeq
- deepinv

### 1.2. Download datasets and pretrained models

We provide a script to download datasets used in PnP-Flow and the corresponding pre-trained networks. The datasets and network checkpoints will be downloaded and stored in the `data` and `model` directories, respectively.

<b>CelebA.</b> To download the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset and the pre-trained OT FM network (U-Net), run the following commands:
```bash
bash download.sh celeba-dataset
bash download.sh pretrained-network-celeba
```

<b>AFHQ-CAT.</b> To download the [AFHQ-CAT](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) dataset and the pre-trained OT FM network (U-Net), run the following commands:
```bash
bash download.sh afhq-cat-dataset
bash download.sh pretrained-network-afhq-cat
```

Note that as the dataset AFHQ-Cat doesn't have a validation split, we create one when downloading the dataset.

Alternatively, the FM models can directly be downloaded here: [CelebA model](https://drive.google.com/file/d/1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6/view?usp=drive_link), [AFHQ-Cat model](https://drive.google.com/file/d/1FpD3cYpgtM8-KJ3Qk48fcjtr1Ne_IMOF/view?usp=drive_link), [MNIST-Dirichlet model](https://drive.google.com/file/d/1If5gkWEfChJHc8v8CCEhGhEeeAqsxKTz/view?usp=drive_link)

And the denoisers for the PnP-GS method here: [CelebA model](https://drive.google.com/file/d/1ZqBeafErEogaXFupW0ZSLL7P9QoRA-lN/view?usp=drive_link), [AFHQ-Cat model](https://drive.google.com/file/d/17AXI9p17c7h_xaI19qDcTT2u9_wu0DQY/view?usp=drive_link)

Disclaimer: The provided CelebA checkpoints correspond to a retrained model and are not the exact ones used in the paper, but results should only vary very slightly.

## 2. Training

You can also use the code to train your own OT Flow Matching model.

You can modify the config options directly in the main_config.yaml file located in ```config/```. Alternatively, config keys can be given as options directly in the command line.

For example, to train the generative flow matching model (here, the U-net is the velocity) on CelebA, with a Gaussian latent distribution, run:
```python
python main.py --opts dataset celeba train True eval False batch_size 128 num_epoch 100
```
At each 5 epochs, the model is saved in ```./model/celeba/gaussian/ot```. Generated samples are saved in ```./results/celeba/gaussian/ot```.

### Computing generative model scores

After the training, the final model is loaded and can be used for generating samples / solving inverse problems. You can compute the full FID (based on 50000 generated samples), the Vendi score, and the Slice Wasserstein score running
```python
python main.py --opts dataset mnist train False eval True compute_metrics True solve_inverse_problem False
```
## 3. Solving inverse problems

The available inverse problems are:
- Denoising --> set ```problem: 'denoising'```
- Gaussian deblurring --> set ```problem: 'gaussian_deblurring'```
- Super-resolution --> set ```problem: 'superresolution'```
- Box inpainting --> set ```problem: 'inpainting'```
- Random inpainting --> set ```problem: 'random_inpainting'```
- Free-form inpainting --> set ```problem: 'paintbrush_inpainting'```

The parameters of the inverse problems (e.g., noise level) can be adjusted manually in the ```main.py``` file.

The available methods are
- ```pnp_flow``` (our method)
- ```ot_ode``` (from this [paper](https://openreview.net/forum?id=PLIt3a4yTm&referrer=%5Bthe%20profile%20of%20Ashwini%20Pokle%5D(%2Fprofile%3Fid%3D~Ashwini_Pokle1)))
- ```d_flow``` (from this [paper](https://arxiv.org/pdf/2402.14017))
- ```flow_priors``` (from this [paper](https://arxiv.org/abs/2405.18816))
- ```pnp_diff``` (from this [paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf))
- ```pnp_gs``` (from this [paper](https://openreview.net/pdf?id=fPhKeld3Okz))

### 3.1. Finding the optimal parameters on the validation set

The optimal parameters can tuned running
```python
python bash scripts/script_val.sh
```

You can also use the optimal values we found, as reported in the Appendix of the paper, and input them into the configuration files of the methods.

### 3.2. Evaluation on the test set

You can either directely run
```python
python main.py --opts dataset celeba train False eval True problem inpainting method pnp_flow
```
or the use the bash file ```scripts/script_test.sh```.

Visual results will be saved in ```results/celeba/inpainting```.

## Acknowledgements
This repository builds upon the following publicly available codes:
- [PnP-GS](https://openreview.net/pdf?id=fPhKeld3Okz) available at https://github.com/samuro95/GSPnP
- [DiffPIR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf) from the [DeepInv](https://deepinv.github.io/deepinv/stubs/deepinv.sampling.DiffPIR.html#deepinv.sampling.DiffPIR) library
- The folder ImageGeneration is copied from [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) repository.
- We thank Yasi Zhang for helping us to debug our Flow Priors code.


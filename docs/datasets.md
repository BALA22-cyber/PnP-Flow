# Microscopy datasets and preprocessing

The source images, derived arrays, split manifests, and model checkpoints are stored separately from this repository. The counts below describe the datasets used in the original experiments; they may change if inputs or preparation options change. See the [README](../README.md) for setup and training commands.

## NFFA-SEM sources and splits

The ten categories are Biological, Fibres, Films/Coated Surface, MEMS devices and electrodes, Nanowires, Particles, Patterned surface, Porous Sponge, Powder, and Tips. The three-class dataset uses Films/Coated Surface, Nanowires, and Particles. Class sizes differ substantially.

Sources were split by category into approximately 80% training, 10% validation, and 10% test sets before patch extraction. A source and its derivatives stay in one split. Preserve the original source manifest to reproduce exact membership; a new split can yield different results.

| Dataset | Preparation per source | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `sem_nffa` | Up to four native 128×128 crops; three classes | 25,820 | 3,220 | 3,236 |
| `sem_nffa_all10` | Up to four native 128×128 crops; ten classes | 68,044 | 8,496 | 8,530 |
| `sem_nffa_all10_512` | Four native 512×512 crops; ten classes | 68,040 | 8,496 | 8,532 |
| `sem_nffa_all10_centerresize_512` | One center-resized 512×512 image | 17,011 | 2,123 | 2,133 |
| `sem_nffa_all10_centerresize_256` | One center-resized 256×256 image | 17,011 | 2,123 | 2,133 |

One validation source in the center-resized set was excluded because its cleaned crop had identical first and 99th percentiles, making normalization undefined. One source was too small for native 512×512 crops. The random-crop datasets can also contain fewer than four accepted patches when a crop lacks sufficient variation.

## Random crops

`scripts/prepare_sem_patches.py` converts each source JPEG to grayscale, normalizes it using the source's first and 99th percentiles, and samples native-resolution crops. It rejects low-variation crops and records source and patch manifests. These images are not resized: the crop retains the source pixel scale. Use `--help` to configure patch size, seed, source split, and output dtype.

## Center crop and resize

`scripts/prepare_nffa_clean_resized.py` reads the supplied source manifest, detects a lower annotation band where possible, extracts the largest centered square from the cleaned grayscale image, and computes first and 99th percentiles on that square. It resizes to 512×512 with Lanczos and normalizes the resized image using those limits to `[-1, 1]`. It saves cleaned and cropped PNGs for inspection and one float16 NPY per accepted source under its existing split and category.

`scripts/prepare_nffa_centerresize_256.py` starts from the cleaned PNGs and processing manifest. It extracts the centered square, calculates the same percentile limits, and resizes directly to 256×256 with Lanczos. It writes normalized float16 NPY files with the same split assignments. These images are resized from the cleaned crop, not from the 512×512 output.

Both center-resize scripts accept paths as command-line options. Use `--help` for required arguments. The output layout used by the loader is `data/<dataset>/{train,val,test}/<category>/*.npy`. The loader converts arrays to float32, adds a channel dimension, and clips values to `[-1, 1]`. Use the matching preprocessing for new inference data.

## HAADF

`scripts/prepare_haadf_patches.py` normalizes an HDF5 frame by percentiles and extracts overlapping patches. Its splits are based on vertical image position, so they are spatial regions of the same source rather than independent source images. See [README_HAADF.md](../README_HAADF.md) for the HDF5 path, usage, and limitations.

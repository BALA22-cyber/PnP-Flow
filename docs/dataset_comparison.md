# Electron microscopy datasets for flow-matching inpainting

A flow-matching prior learns the image distribution it is trained on. Clean, unmasked images suffice for the unconditional prior in this repository; synthetic masks can be applied during inpainting evaluation. The imaging mode, scale, and structures still matter when applying a prior to a new domain.

| Dataset or source | Potential use | Main consideration |
| --- | --- | --- |
| Target HAADF-STEM images | Target-domain training, fine-tuning, and evaluation | Limited numbers of source images and correlated patches can restrict generalization. |
| NFFA-SEM | Diverse SEM surface and nanomaterial images for training and pretraining | SEM texture and scale differ from atomic-resolution HAADF-STEM. |
| [EPFL electron microscopy dataset](https://www.epfl.ch/labs/cvlab/data/data-em/) | Controlled biological EM pipeline experiments | Biological tissue and adjacent volume slices differ from HAADF; split correlated volumes carefully. |
| [EMPIAR](https://www.ebi.ac.uk/empiar/) | Finding additional public microscopy data | It is an archive of varied modalities; select entries matching the intended task and check their metadata and licenses. |

A practical evaluation uses source-level held-out SEM images for SEM results and independently held-out HAADF sources when available for target-domain results. For data from a single frame, a spatial split is more informative than a random patch split but does not establish generalization to independent frames. Report mask shape and size, missing-region metrics, and preprocessing. A plausible reconstruction is not necessarily the true missing structure.

See [datasets.md](datasets.md) for the datasets actually prepared by this project and [inpainting_workflows.md](inpainting_workflows.md) for the implemented workflows. The public dataset descriptions and availability should be checked at the linked sources before use.

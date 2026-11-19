#!/usr/bin/env python
"""Compare all-class and single-class 512 px NFFA priors on identical box masks."""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pnpflow.utils import define_model


class ModelArgs:
    model = "ot"
    num_channels = 1
    dim_image = 512


@torch.inference_mode()
def pnp_flow_inpaint(model, clean, missing, steps, noise_std, lr_pnp, alpha, num_samples, seed):
    torch.manual_seed(seed)
    if clean.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    known = (~missing).to(dtype=clean.dtype)
    observed = known * clean
    x = observed.clone()
    lr = noise_std**2 * lr_pnp

    for iteration in range(steps):
        t = torch.full((1,), iteration / steps, device=clean.device, dtype=clean.dtype)
        t_view = t.view(1, 1, 1, 1)
        lr_t = lr * (1.0 - t).pow(alpha).view(1, 1, 1, 1)
        data_gradient = known * (known * x - observed) / (noise_std**2)
        z = x - lr_t * data_gradient

        z_samples = z.expand(num_samples, -1, -1, -1)
        t_samples = t.expand(num_samples)
        t_samples_view = t_samples.view(num_samples, 1, 1, 1)
        z_tilde = t_samples_view * z_samples + (1.0 - t_samples_view) * torch.randn_like(z_samples)
        velocity = model(z_tilde, t_samples)
        candidates = z_tilde + (1.0 - t_samples_view) * velocity
        x_new = candidates.mean(dim=0, keepdim=True)
        x = known * observed + (1.0 - known) * x_new

    return observed, x.clamp(-1.0, 1.0)


def load_prior(checkpoint_path, device):
    model, _ = define_model(ModelArgs())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError(f"Checkpoint has non-finite parameters: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default="sem_nffa_all10_centerresize_512")
    parser.add_argument("--class-name", default="Patterned_surface")
    parser.add_argument("--all-class-run", required=True)
    parser.add_argument("--single-class-run", required=True)
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--mask-sizes", default="64,128,256")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--lr-pnp", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.steps < 1 or args.num_samples < 1 or args.noise_std <= 0:
        raise ValueError("steps, num_samples, and noise_std must be positive.")
    root = args.repo_root.resolve()
    os.chdir(root)
    device = torch.device(args.device)
    dataset_root = root / "data" / args.dataset
    test_files = sorted((dataset_root / "test" / args.class_name).glob("*.npy"))
    if not test_files:
        raise FileNotFoundError(f"No {args.class_name} test patches under {dataset_root / 'test'}")
    if not 0 <= args.test_index < len(test_files):
        raise IndexError(f"test-index {args.test_index} is outside 0..{len(test_files) - 1}")

    run_paths = {
        "all_class": root / "model" / args.dataset / "ot" / args.all_class_run,
        "single_class": root / "model" / args.dataset / "ot" / args.single_class_run,
    }
    run_paths = {
        name: (directory / "model_final.pt" if (directory / "model_final.pt").is_file()
               else directory / "training_state_latest.pt")
        for name, directory in run_paths.items()
    }
    for name, path in run_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} final and resumable checkpoints under {path.parent}")
    output_dir = root / "notebooks" / "outputs" / f"{args.single_class_run}_{args.class_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_path = test_files[args.test_index]
    patch = np.load(patch_path).astype(np.float32)
    if patch.ndim == 3 and patch.shape[0] == 1:
        patch = patch[0]
    if patch.shape != (512, 512) or not np.isfinite(patch).all():
        raise ValueError(f"Expected a finite 512 x 512 test patch, got {patch.shape} at {patch_path}")
    clean = torch.from_numpy(np.clip(patch, -1.0, 1.0))[None, None].to(device)

    priors = {name: load_prior(path, device) for name, path in run_paths.items()}
    mask_sizes = [int(value) for value in args.mask_sizes.split(",") if value.strip()]
    results = []
    panels = []
    for size in mask_sizes:
        if not 0 < size < 512:
            raise ValueError(f"Mask size must be between 1 and 511, got {size}")
        top = (512 - size) // 2
        left = (512 - size) // 2
        missing = torch.zeros_like(clean, dtype=torch.bool)
        missing[:, :, top:top + size, left:left + size] = True
        row = {"size": size, "top": top, "left": left}
        outputs = {}

        for model_index, (name, model) in enumerate(priors.items()):
            start = time.perf_counter()
            observed, reconstruction = pnp_flow_inpaint(
                model, clean, missing, args.steps, args.noise_std, args.lr_pnp,
                args.alpha, args.num_samples, args.seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            recon_np = reconstruction[0, 0].cpu().numpy()
            clean_np = clean[0, 0].cpu().numpy()
            missing_np = missing[0, 0].cpu().numpy()
            error = recon_np - clean_np
            masked_mse = float(np.mean(error[missing_np] ** 2))
            baseline_mse = float(np.mean(clean_np[missing_np] ** 2))
            row[name] = {
                "masked_psnr_db": float(10 * np.log10(4.0 / max(masked_mse, 1e-12))),
                "zero_fill_baseline_psnr_db": float(10 * np.log10(4.0 / max(baseline_mse, 1e-12))),
                "max_known_pixel_change": float(np.max(np.abs(error[~missing_np]))),
                "seconds": elapsed,
            }
            outputs[name] = recon_np
            np.save(
                root / "notebooks" / "outputs" / f"{args.single_class_run}_{args.class_name}"
                / f"reconstruction_{name}_mask{size}.npy",
                recon_np.astype(np.float32),
            )

        delta = np.where(missing_np, np.abs(outputs["single_class"] - outputs["all_class"]), np.nan)
        panels.append((size, clean_np, observed[0, 0].cpu().numpy(), missing_np,
                       outputs["all_class"], outputs["single_class"], delta))
        results.append(row)

    fig, axes = plt.subplots(len(panels), 6, figsize=(22, 4.2 * len(panels)), squeeze=False,
                             constrained_layout=True)
    for row_index, (size, clean_np, observed_np, mask_np, all_np, single_np, delta_np) in enumerate(panels):
        psnr_all = results[row_index]["all_class"]["masked_psnr_db"]
        psnr_single = results[row_index]["single_class"]["masked_psnr_db"]
        row_images = [
            (clean_np, "Clean test patch", "gray", -1, 1),
            (observed_np, "Observed input", "gray", -1, 1),
            (mask_np, f"Missing mask\n{size} x {size}", "gray", 0, 1),
            (all_np, f"All-class\nPSNR {psnr_all:.2f} dB", "gray", -1, 1),
            (single_np, f"{args.class_name}\nPSNR {psnr_single:.2f} dB", "gray", -1, 1),
            (delta_np, "Difference in mask", "magma", 0, max(float(np.nanmax(delta_np)), 1e-6)),
        ]
        for ax, (image, title, cmap, vmin, vmax) in zip(axes[row_index], row_images):
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")

    fig.suptitle(f"NFFA {args.class_name}: all-class vs single-class PnP-Flow", fontsize=15)
    figure_path = output_dir / "inpainting_comparison.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    metrics_path = output_dir / "inpainting_metrics.json"
    metrics_path.write_text(json.dumps({
        "class_name": args.class_name,
        "test_patch": str(patch_path),
        "checkpoints": {name: str(path) for name, path in run_paths.items()},
        "steps": args.steps,
        "noise_std": args.noise_std,
        "lr_pnp": args.lr_pnp,
        "alpha": args.alpha,
        "num_samples": args.num_samples,
        "results": results,
    }, indent=2) + "\n")

    print(f"Test patch: {patch_path}")
    for result in results:
        print(
            f"mask={result['size']}x{result['size']}: "
            f"all-class PSNR={result['all_class']['masked_psnr_db']:.2f} dB; "
            f"single-class PSNR={result['single_class']['masked_psnr_db']:.2f} dB"
        )
    print(f"Comparison figure: {figure_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()

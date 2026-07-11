#!/usr/bin/env python3
"""Render a reproducible grid of prepared SEM patches for visual inspection.

Example:
  python scripts/visualize_sem_patches.py \
    --dataset-root data/sem_nffa \
    --split train \
    --count 36 \
    --output results/sem_nffa/train_patches.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np


def read_categories(dataset_root: Path) -> dict[str, str]:
    manifest = dataset_root / "patch_manifest.csv"
    if not manifest.is_file():
        return {}

    with manifest.open(newline="", encoding="utf-8") as handle:
        return {
            row["patch"]: row["category"]
            for row in csv.DictReader(handle)
            if row.get("patch") and row.get("category")
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize prepared SEM .npy patches.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/sem_nffa"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Open the figure after saving it.")
    args = parser.parse_args()

    if args.count <= 0 or args.columns <= 0:
        raise ValueError("count and columns must be positive.")

    split_root = args.dataset_root / args.split
    patches = sorted(split_root.glob("*.npy"))
    if not patches:
        raise FileNotFoundError(f"No .npy patches found in {split_root}")

    count = min(args.count, len(patches))
    indices = np.random.default_rng(args.seed).choice(len(patches), size=count, replace=False)
    selected = [patches[index] for index in sorted(indices)]
    categories = read_categories(args.dataset_root)

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(count / args.columns))
    figure, axes = plt.subplots(rows, args.columns, figsize=(2.5 * args.columns, 2.8 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for axis, patch_path in zip(flat_axes, selected):
        patch = np.load(patch_path)
        if patch.ndim != 2:
            raise ValueError(f"Expected a 2D patch, got {patch.shape} in {patch_path}")
        if not np.isfinite(patch).all():
            raise ValueError(f"Patch contains non-finite values: {patch_path}")

        category = categories.get(f"{args.split}/{patch_path.name}", "unknown")
        axis.imshow(patch, cmap="gray", vmin=-1.0, vmax=1.0, interpolation="nearest")
        axis.set_title(f"{category}\n{patch_path.stem[-15:]}", fontsize=8)
        axis.axis("off")

    for axis in flat_axes[count:]:
        axis.axis("off")

    figure.suptitle(
        f"SEM patches: {args.split} ({count} sampled, fixed grayscale range [-1, 1])",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    output = args.output or Path("results") / "sem_nffa" / f"patches_{args.split}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"Saved {count} patch previews to {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

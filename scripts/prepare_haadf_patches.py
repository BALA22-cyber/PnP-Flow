#!/usr/bin/env python3
"""Prepare HAADF HDF5 data as normalized .npy patches for PnP-Flow.

Example:
  python scripts/prepare_haadf_patches.py \
    --h5 data/HAADF_21.h5 \
    --out data/haadf2 \
    --patch-size 128 \
    --stride 64
"""
from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np


def normalize_haadf(im: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> tuple[np.ndarray, dict]:
    im = im.astype(np.float32)
    lo, hi = np.percentile(im, [p_low, p_high])
    clipped = np.clip(im, lo, hi)
    norm = 2.0 * (clipped - lo) / max(hi - lo, 1e-8) - 1.0
    return norm.astype(np.float32), {"lo": float(lo), "hi": float(hi), "p_low": p_low, "p_high": p_high}


def extract_patches(im: np.ndarray, patch_size: int, stride: int):
    patches, coords = [], []
    h, w = im.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(im[y:y + patch_size, x:x + patch_size])
            coords.append((y, x))
    return np.stack(patches), coords


def save_split(patches: np.ndarray, coords: list[tuple[int, int]], out: Path, image_h: int):
    split_dirs = {name: out / name for name in ["train", "val", "test"]}
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.npy"):
            f.unlink()

    counts = {"train": 0, "val": 0, "test": 0}
    for i, (patch, (y, x)) in enumerate(zip(patches, coords)):
        frac_y = y / max(image_h - 1, 1)
        if frac_y < 0.70:
            split = "train"
        elif frac_y < 0.85:
            split = "val"
        else:
            split = "test"
        fname = split_dirs[split] / f"patch_{i:05d}_y{y:04d}_x{x:04d}.npy"
        np.save(fname, patch.astype(np.float32))
        counts[split] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to HAADF .h5 file")
    ap.add_argument("--out", default="data/haadf2", help="Output dataset directory")
    ap.add_argument("--dataset-path", default="Measurement_000/Channel_000/HAADF/HAADF")
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--p-low", type=float, default=1.0)
    ap.add_argument("--p-high", type=float, default=99.0)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        im = f[args.dataset_path][()]

    norm, stats = normalize_haadf(im, args.p_low, args.p_high)
    patches, coords = extract_patches(norm, args.patch_size, args.stride)
    counts = save_split(patches, coords, Path(args.out), norm.shape[0])

    print("Loaded:", args.h5)
    print("Dataset path:", args.dataset_path)
    print("Original shape:", im.shape, "dtype:", im.dtype)
    print("Normalization:", stats)
    print("Patch array:", patches.shape)
    print("Saved counts:", counts)
    print("Output:", args.out)


if __name__ == "__main__":
    main()

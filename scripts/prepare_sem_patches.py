#!/usr/bin/env python3
"""Convert category-organized SEM JPEGs into PnP-Flow grayscale .npy patches.

Example:
  python scripts/prepare_sem_patches.py \
    --input-root /data2/bselvaku/datasets/nffa_sem/raw_images \
    --output-root /data2/bselvaku/datasets/nffa_sem/patches_128
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg"}
SPLITS = ("train", "val", "test")


def category_for(path: Path) -> str:
    return path.parts[0] if len(path.parts) > 1 else "uncategorized"


def source_id(path: Path) -> str:
    return hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()[:12]


def split_sources(paths: list[Path], seed: int, train_fraction: float, val_fraction: float) -> dict[str, list[Path]]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must be in (0, 1) and sum to less than 1.")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[category_for(path)].append(path)

    rng = np.random.default_rng(seed)
    splits = {split: [] for split in SPLITS}
    for category in sorted(grouped):
        items = sorted(grouped[category])
        items = [items[index] for index in rng.permutation(len(items))]
        train_end = int(len(items) * train_fraction)
        val_end = train_end + int(len(items) * val_fraction)
        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])
    return splits


def load_normalized(path: Path, p_low: float, p_high: float) -> tuple[np.ndarray, float, float] | None:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(pixels, [p_low, p_high])
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-6:
        return None
    pixels = np.clip(pixels, low, high)
    pixels = 2.0 * (pixels - low) / (high - low) - 1.0
    return pixels.astype(np.float32), float(low), float(high)


def prepare_output(root: Path, overwrite: bool) -> None:
    existing = [path for split in SPLITS for path in (root / split).glob("*.npy")]
    if existing and not overwrite:
        raise FileExistsError(f"{root} already contains patches. Pass --overwrite to replace this derived dataset.")
    for split in SPLITS:
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in directory.glob("*.npy"):
                path.unlink()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SEM JPEG patches for PnP-Flow.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.patch_size <= 0 or args.patches_per_image <= 0:
        raise ValueError("patch_size and patches_per_image must be positive.")
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    paths = sorted(
        path.relative_to(args.input_root)
        for path in args.input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No JPEG images found below {args.input_root}")

    prepare_output(args.output_root, args.overwrite)
    source_splits = split_sources(paths, args.seed, args.train_fraction, args.val_fraction)
    rng = np.random.default_rng(args.seed)
    counts = {split: 0 for split in SPLITS}
    source_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for split in SPLITS:
        for relative_path in source_splits[split]:
            source_path = args.input_root / relative_path
            category = category_for(relative_path)
            try:
                loaded = load_normalized(source_path, args.p_low, args.p_high)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                skipped_rows.append({"source": relative_path.as_posix(), "reason": f"load_error: {error}"})
                continue
            if loaded is None:
                skipped_rows.append({"source": relative_path.as_posix(), "reason": "constant_or_invalid_image"})
                continue

            image, low, high = loaded
            height, width = image.shape
            source_rows.append({
                "source": relative_path.as_posix(),
                "category": category,
                "split": split,
                "height": height,
                "width": width,
                "p_low_value": low,
                "p_high_value": high,
            })
            if height < args.patch_size or width < args.patch_size:
                skipped_rows.append({"source": relative_path.as_posix(), "reason": "image_smaller_than_patch"})
                continue

            accepted = 0
            for _ in range(args.patches_per_image * 8):
                if accepted >= args.patches_per_image:
                    break
                y = int(rng.integers(0, height - args.patch_size + 1))
                x = int(rng.integers(0, width - args.patch_size + 1))
                patch = image[y:y + args.patch_size, x:x + args.patch_size]
                if float(patch.std()) < args.min_std:
                    continue
                patch_name = f"{category}_{source_id(relative_path)}_{accepted:02d}.npy"
                np.save(args.output_root / split / patch_name, patch.astype(np.float32))
                patch_rows.append({
                    "patch": f"{split}/{patch_name}",
                    "source": relative_path.as_posix(),
                    "category": category,
                    "split": split,
                    "x": x,
                    "y": y,
                    "patch_size": args.patch_size,
                })
                counts[split] += 1
                accepted += 1
            if accepted < args.patches_per_image:
                skipped_rows.append({"source": relative_path.as_posix(), "reason": f"only_{accepted}_patches_above_min_std"})

    write_csv(args.output_root / "source_manifest.csv", ["source", "category", "split", "height", "width", "p_low_value", "p_high_value"], source_rows)
    write_csv(args.output_root / "patch_manifest.csv", ["patch", "source", "category", "split", "x", "y", "patch_size"], patch_rows)
    write_csv(args.output_root / "skipped_images.csv", ["source", "reason"], skipped_rows)

    print(f"Input images: {len(paths)}")
    print(f"Source split counts: { {split: len(source_splits[split]) for split in SPLITS} }")
    print(f"Patch counts: {counts}")
    print(f"Skipped records: {len(skipped_rows)}")
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()

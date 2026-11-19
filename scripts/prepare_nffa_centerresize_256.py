#!/usr/bin/env python3
"""Build the 256x256 center-resized NFFA-SEM dataset from cleaned PNGs.

The source processing manifest supplies the original source-level split. For
each entry, this script reads its footer-cleaned grayscale PNG, extracts the
largest centered square, computes the 1st/99th percentile limits on that crop,
resizes directly to 256x256 with Lanczos, and writes one normalized float16
NPY under train/, val/, or test/.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_nffa_clean_resized import center_square, normalize_for_model, percentile_range


SPLITS = {"train", "val", "test"}
TARGET_SIZE = 256
DEFAULT_CLEAN_ROOT = Path("/data2/marci/bselvaku/nffa_sem/clean")


def is_valid_npy(path: Path) -> bool:
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        return (
            array.shape == (TARGET_SIZE, TARGET_SIZE)
            and array.dtype == np.float16
            and np.isfinite(array).all()
            and float(array.min()) >= -1.0001
            and float(array.max()) <= 1.0001
        )
    except Exception:
        return False


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array, dtype=np.float16), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def process_one(record: dict[str, str], clean_root: Path, output_root: Path, p_low: float,
                p_high: float, overwrite: bool) -> dict[str, object]:
    source = Path(record["source"])
    category = record["category"]
    split = record["split"]
    if split not in SPLITS:
        raise ValueError(f"invalid split {split!r} for {source}")
    if Path(category).name != category:
        raise ValueError(f"invalid category directory {category!r}")

    clean_png = clean_root / category / f"{source.stem}.png"
    output_npy = output_root / split / category / f"{source.stem}.npy"
    if not clean_png.is_file():
        raise FileNotFoundError(f"cleaned source image not found: {clean_png}")

    with Image.open(clean_png) as handle:
        cleaned = np.asarray(handle.convert("L"), dtype=np.uint8)
    if cleaned.ndim != 2 or min(cleaned.shape) < 2:
        raise ValueError(f"expected a valid grayscale image, got {cleaned.shape}")

    cropped, crop_box = center_square(cleaned)
    low, high = percentile_range(cropped, p_low, p_high)
    if not overwrite and is_valid_npy(output_npy):
        status = "reused"
    else:
        resized = np.asarray(
            Image.fromarray(cropped, mode="L").resize(
                (TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
        normalized = normalize_for_model(resized, low, high)
        atomic_save_npy(output_npy, normalized)
        if not is_valid_npy(output_npy):
            raise IOError(f"output validation failed: {output_npy}")
        status = "processed"

    return {
        "source": source.as_posix(),
        "category": category,
        "split": split,
        "input_clean_png": str(clean_png),
        "output_npy": str(output_npy),
        "crop_top": crop_box[0],
        "crop_left": crop_box[1],
        "crop_bottom": crop_box[2],
        "crop_right": crop_box[3],
        "crop_size": cropped.shape[0],
        "target_size": TARGET_SIZE,
        "p_low_value": low,
        "p_high_value": high,
        "dtype": "float16",
        "status": status,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data" / "sem_nffa_all10_centerresize_256",
    )
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.source_manifest or args.clean_root / "processing_manifest.csv"
    if not args.clean_root.is_dir():
        raise FileNotFoundError(args.clean_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not 0 <= args.p_low < args.p_high <= 100:
        raise ValueError("percentiles must satisfy 0 <= p-low < p-high <= 100")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)
        missing = {"source", "category", "split"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"source manifest is missing columns: {sorted(missing)}")
    if not records:
        raise ValueError(f"source manifest is empty: {manifest_path}")

    print(f"Clean image root: {args.clean_root}", flush=True)
    print(f"Source manifest: {manifest_path}", flush=True)
    print(f"Output dataset: {args.output_root}", flush=True)
    print(f"Records: {len(records)}; target: {TARGET_SIZE}x{TARGET_SIZE}; workers: {args.workers}", flush=True)

    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    counts: Counter[tuple[str, str]] = Counter()

    def run(record: dict[str, str]) -> dict[str, object]:
        try:
            return process_one(record, args.clean_root, args.output_root,
                               args.p_low, args.p_high, args.overwrite)
        except Exception as exc:
            return {
                "source": record.get("source", ""),
                "category": record.get("category", ""),
                "split": record.get("split", ""),
                "status": "failed",
                "error": repr(exc),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(run, records), start=1):
            if result["status"] == "failed":
                failures.append(result)
            else:
                successes.append(result)
                counts[(str(result["split"]), str(result["status"]))] += 1
            if index % 500 == 0 or index == len(records):
                print(f"Processed {index}/{len(records)} records", flush=True)

    output_fields = [
        "source", "category", "split", "input_clean_png", "output_npy",
        "crop_top", "crop_left", "crop_bottom", "crop_right", "crop_size",
        "target_size", "p_low_value", "p_high_value", "dtype", "status",
    ]
    write_csv(args.output_root / "processing_manifest.csv", successes, output_fields)
    write_csv(
        args.output_root / "processing_failures.csv",
        failures,
        ["source", "category", "split", "status", "error"],
    )
    print(f"Split/status counts: {dict(sorted(counts.items()))}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)
    print(f"Dataset ready at: {args.output_root}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

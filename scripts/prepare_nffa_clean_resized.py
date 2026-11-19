#!/usr/bin/env python3
"""Prepare the NFFA SEM all-class center-crop/resize training dataset.

For each source listed in the source manifest this script writes four separate
artifacts:

  clean/<category>/<name>.png
  centercrop/<category>/<name>.png
  centercrop/resized/images/<category>/<name>.png
  centercrop/resized/processed/<split>/<category>/<name>.npy

The NPY file is a finite float16 HxW tensor normalized to [-1, 1]. The
source-level split in the supplied manifest is authoritative; the script does
not create a new split.

Without --overwrite, complete records are validated and reused, while
incomplete or invalid records are rebuilt. With --overwrite, selected records
are rebuilt. Failures are recorded separately.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

SPLITS = ("train", "val", "test")


def detect_annotation_band(
    pixels: np.ndarray,
    bright_threshold: float = 180.0,
    bright_fraction: float = 0.50,
    min_band_rows: int = 8,
) -> tuple[int, int, str]:
    """Return content-bottom, band-bottom, and the detection rule.

    The microscopy content itself can be bright, so a threshold applied from
    row zero can incorrectly classify the upper image as the footer.  The
    annotation panel is a sustained, much brighter horizontal region in the
    lower part of the frame; require both high mean intensity and a high bright
    pixel fraction there.
    """
    height = pixels.shape[0]
    row_mean = pixels.astype(np.float32).mean(axis=1)
    bright_fraction_by_row = (pixels >= bright_threshold).mean(axis=1)
    candidate = (row_mean >= 205.0) & (bright_fraction_by_row >= 0.70)
    run_start: int | None = None
    gaps = 0
    for row in range(max(0, int(height * 0.55)), height):
        is_candidate = bool(candidate[row])
        if is_candidate:
            if run_start is None:
                run_start = row
            gaps = 0
        elif run_start is not None:
            gaps += 1
            if gaps > 1:
                if row - gaps - run_start >= min_band_rows:
                    return run_start, height, "bright_lower_annotation_band"
                run_start = None
                gaps = 0
    if run_start is not None and height - run_start >= min_band_rows:
        return run_start, height, "bright_lower_annotation_band"
    return pixels.shape[0], pixels.shape[0], "no_annotation_band_detected"


def center_square(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top : top + side, left : left + side], (top, left, top + side, left + side)


def percentile_range(image: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    low, high = np.percentile(image.astype(np.float32), [p_low, p_high])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"invalid percentile range [{low}, {high}]")
    return float(low), float(high)


def normalize_for_model(image: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = 2.0 * np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0) - 1.0
    if not np.isfinite(normalized).all():
        raise ValueError("normalization produced non-finite values")
    return normalized.astype(np.float16)


def atomic_save_png(path: Path, image: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npy(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        np.save(temporary, np.asarray(array, dtype=np.float16), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def image_is_valid(path: Path, expected_shape: tuple[int, int] | None = None) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            if expected_shape is not None and image.size != (expected_shape[1], expected_shape[0]):
                return False
        return True
    except Exception:
        return False


def npy_is_valid(path: Path, target_size: int) -> bool:
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        return (
            array.shape == (target_size, target_size)
            and array.dtype == np.float16
            and np.isfinite(array).all()
            and float(array.min()) >= -1.0001
            and float(array.max()) <= 1.0001
        )
    except Exception:
        return False


def output_paths(output_root: Path, category: str, name: str, split: str) -> dict[str, Path]:
    filename = Path(name).with_suffix(".png").name
    stem = Path(filename).stem
    return {
        "clean_png": output_root / category / filename,
        "centercrop_png": output_root / "centercrop" / category / filename,
        "resized_png": output_root / "centercrop" / "resized" / "images" / category / filename,
        "resized_npy": output_root / "centercrop" / "resized" / "processed" / split / category / f"{stem}.npy",
    }


def complete_outputs(paths: dict[str, Path], target_size: int) -> bool:
    return (
        image_is_valid(paths["clean_png"])
        and image_is_valid(paths["centercrop_png"])
        and image_is_valid(paths["resized_png"], (target_size, target_size))
        and npy_is_valid(paths["resized_npy"], target_size)
    )


def process_one(
    record: dict[str, str],
    input_root: Path,
    output_root: Path,
    target_size: int,
    p_low: float,
    p_high: float,
    overwrite: bool,
) -> dict[str, object]:
    source_relative = Path(record["source"])
    category = record["category"]
    split = record["split"]
    if split not in SPLITS:
        raise ValueError(f"invalid split {split!r} for {source_relative}")
    source = input_root / source_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    paths = output_paths(output_root, category, source_relative.name, split)
    if not overwrite and complete_outputs(paths, target_size):
        with Image.open(source) as handle:
            original = np.asarray(handle.convert("L"), dtype=np.uint8)
        content_bottom, band_bottom, band_rule = detect_annotation_band(original)
        cleaned = original[:content_bottom]
        cropped, crop_box = center_square(cleaned)
        low, high = percentile_range(cropped, p_low, p_high)
        result = dict(record)
        result.update(
            {
                "source_path": str(source),
                "original_height": original.shape[0],
                "original_width": original.shape[1],
                "content_bottom_row_exclusive": content_bottom,
                "annotation_band_bottom_row_exclusive": band_bottom,
                "removed_rows": original.shape[0] - content_bottom,
                "annotation_rule": band_rule,
                "crop_top": crop_box[0],
                "crop_left": crop_box[1],
                "crop_bottom": crop_box[2],
                "crop_right": crop_box[3],
                "clean_shape": f"{cleaned.shape[0]}x{cleaned.shape[1]}",
                "centercrop_shape": f"{cropped.shape[0]}x{cropped.shape[1]}",
                "resized_shape": f"{target_size}x{target_size}",
                "p_low_value": low,
                "p_high_value": high,
                "status": "reused",
            }
        )
        result.update({key: str(value) for key, value in paths.items()})
        return result

    with Image.open(source) as handle:
        original = np.asarray(handle.convert("L"), dtype=np.uint8)
    if original.ndim != 2 or min(original.shape) < 2:
        raise ValueError(f"expected a valid 2D image, got {original.shape}")
    content_bottom, band_bottom, band_rule = detect_annotation_band(original)
    cleaned = original[:content_bottom]
    if min(cleaned.shape) < 2:
        raise ValueError(f"annotation removal left unusable shape {cleaned.shape}")
    cropped, crop_box = center_square(cleaned)
    low, high = percentile_range(cropped, p_low, p_high)
    resized = np.asarray(
        Image.fromarray(cropped, mode="L").resize((target_size, target_size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    normalized = normalize_for_model(resized, low, high)

    atomic_save_png(paths["clean_png"], cleaned, overwrite)
    atomic_save_png(paths["centercrop_png"], cropped, overwrite)
    atomic_save_png(paths["resized_png"], resized, overwrite)
    atomic_save_npy(paths["resized_npy"], normalized, overwrite)
    if not complete_outputs(paths, target_size):
        raise IOError(f"output validation failed for {source_relative}")

    result = dict(record)
    result.update(
        {
            "status": "processed",
            "source_path": str(source),
            "original_height": original.shape[0],
            "original_width": original.shape[1],
            "content_bottom_row_exclusive": content_bottom,
            "annotation_band_bottom_row_exclusive": band_bottom,
            "removed_rows": original.shape[0] - content_bottom,
            "annotation_rule": band_rule,
            "crop_top": crop_box[0],
            "crop_left": crop_box[1],
            "crop_bottom": crop_box[2],
            "crop_right": crop_box[3],
            "clean_shape": f"{cleaned.shape[0]}x{cleaned.shape[1]}",
            "centercrop_shape": f"{cropped.shape[0]}x{cropped.shape[1]}",
            "resized_shape": f"{target_size}x{target_size}",
            "p_low_value": low,
            "p_high_value": high,
        }
    )
    result.update({key: str(value) for key, value in paths.items()})
    return result


def fit_preview(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    preview = ImageOps.contain(image.convert("RGB"), (size[0] - 8, size[1] - 22), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(preview, ((size[0] - preview.width) // 2, 18 + (size[1] - 22 - preview.height) // 2))
    return canvas


def write_contact(rows: list[dict[str, object]], output: Path, seed: int = 0) -> None:
    """Write a compact QA sheet with two samples from every category."""
    if not rows:
        return
    rng = np.random.default_rng(seed)
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    selected: list[dict[str, object]] = []
    for category in sorted(by_category):
        candidates = by_category[category]
        selected.append(candidates[0])
        if len(candidates) > 1:
            selected.append(candidates[int(rng.integers(0, len(candidates)))])
    width, height = 220, 170
    sheet = Image.new("RGB", (width * 4, (height + 30) * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    columns = ["original", "annotation removed", "center crop", "512x512 resized"]
    for index, row in enumerate(selected):
        y = index * (height + 30)
        draw.text((4, y + 2), f"{row['category']} | {row['source']} | {row.get('original_height','?')}x{row.get('original_width','?')}", fill="black")
        sources = [
            Path(str(row.get("source_path", ""))),
            Path(str(row["clean_png"])),
            Path(str(row["centercrop_png"])),
            Path(str(row["resized_png"])),
        ]
        for col, (title, source) in enumerate(zip(columns, sources)):
            x = col * width
            draw.text((x + 4, y + 16), title, fill="black")
            with Image.open(source) as image:
                panel = fit_preview(image, (width - 8, height - 22))
            sheet.paste(panel, (x + 4, y + 38))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source", "category", "split"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"source manifest missing columns: {sorted(missing)}")
    if len({row["source"] for row in rows}) != len(rows):
        raise ValueError("source manifest contains duplicate source paths")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5, help="Number of manifest records; 0 means all.")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent image workers; 1 preserves sequential processing.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild selected records even when outputs are valid.")
    args = parser.parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    records = read_records(args.source_manifest)
    if args.limit > 0:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for record in records:
            grouped[record["category"]].append(record)
        selected: list[dict[str, str]] = []
        for category in sorted(grouped):
            if len(selected) >= args.limit:
                break
            selected.append(grouped[category][0])
        records = selected[: args.limit]

    def run_record(record: dict[str, str]) -> dict[str, object]:
        try:
            return process_one(record, args.input_root, args.output_root, args.target_size, args.p_low, args.p_high, args.overwrite)
        except Exception as exc:
            return {**record, "status": "failed", "error": repr(exc)}

    successful: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # executor.map preserves input order even though work completes out of order.
        results = executor.map(run_record, records)
        for index, result in enumerate(results, start=1):
            if result.get("status") == "failed":
                failures.append(result)
            else:
                successful.append(result)
            if index % 250 == 0 or index == len(records):
                print(f"Processed {index}/{len(records)} records; successes={len(successful)} failures={len(failures)}", flush=True)

    fields = sorted({key for row in successful for key in row})
    write_csv(args.output_root / "processing_manifest.csv", successful, fields or ["source", "category", "split", "status"])
    failure_fields = sorted({key for row in failures for key in row})
    if failures:
        write_csv(args.output_root / "processing_failures.csv", failures, failure_fields)
    elif (args.output_root / "processing_failures.csv").exists():
        (args.output_root / "processing_failures.csv").unlink()
    write_contact(successful, args.contact_sheet)
    print(f"Selected records: {len(records)}")
    print(f"Successful records: {len(successful)} ({Counter(row['status'] for row in successful)})")
    print(f"Failed records: {len(failures)}")
    print(f"Manifest: {args.output_root / 'processing_manifest.csv'}")
    print(f"Contact sheet: {args.contact_sheet}")


if __name__ == "__main__":
    main()

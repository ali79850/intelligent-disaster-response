"""
Phase 4, Baseline 2: Extract hand-crafted per-building features comparing
pre- and post-disaster imagery, for classical ML damage classification.

Processes tile-by-tile (loading each image pair once, computing Sobel edge
maps once per tile rather than per building) for efficiency. Writes
incrementally to CSV so a long run isn't lost to interruption. Un-classified
buildings are skipped (no reliable label).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, ".")
from src.preprocessing.polygon_utils import get_buildings_for_tile

IMAGES_DIR = Path("data/raw/xbd/train/images")
LABELS_DIR = Path("data/raw/xbd/train/labels")
SPLITS_DIR = Path("data/interim/splits")
OUTPUT_DIR = Path("data/processed/features")

FEATURE_NAMES = [
    "mean_diff", "std_diff", "mean_pre", "mean_post", "area_pixels",
    "edge_density_pre", "edge_density_post", "edge_density_change",
]


def compute_edge_map(gray_img: np.ndarray) -> np.ndarray:
    """Computes a binary edge map for the WHOLE image once - never call this
    per-building, that was the bug causing the extraction to hang."""
    sx = ndimage.sobel(gray_img, axis=0)
    sy = ndimage.sobel(gray_img, axis=1)
    edges = np.hypot(sx, sy)
    return edges > np.percentile(edges, 90)


def edge_density_in_mask(edge_map: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    return float((edge_map & mask).sum()) / float(mask.sum())


def extract_features_for_tile(base_name: str) -> list[dict]:
    pre_path = IMAGES_DIR / f"{base_name}_pre_disaster.png"
    post_path = IMAGES_DIR / f"{base_name}_post_disaster.png"
    label_path = LABELS_DIR / f"{base_name}_post_disaster.json"

    pre_gray = np.array(Image.open(pre_path).convert("L"), dtype=np.float32)
    post_gray = np.array(Image.open(post_path).convert("L"), dtype=np.float32)
    diff = np.abs(post_gray - pre_gray)

    # Compute each expensive image-level operation ONCE per tile, not per building
    pre_edges = compute_edge_map(pre_gray)
    post_edges = compute_edge_map(post_gray)

    buildings = get_buildings_for_tile(label_path)
    rows = []
    for b in buildings:
        if b["subtype"] == "un-classified" or b["subtype"] is None:
            continue

        mask = b["polygon_mask"]
        if mask.sum() == 0:
            continue

        row = {
            "base_name": base_name,
            "uid": b["uid"],
            "subtype": b["subtype"],
            "mean_diff": float(diff[mask].mean()),
            "std_diff": float(diff[mask].std()),
            "mean_pre": float(pre_gray[mask].mean()),
            "mean_post": float(post_gray[mask].mean()),
            "area_pixels": int(mask.sum()),
            "edge_density_pre": edge_density_in_mask(pre_edges, mask),
            "edge_density_post": edge_density_in_mask(post_edges, mask),
        }
        row["edge_density_change"] = row["edge_density_post"] - row["edge_density_pre"]
        rows.append(row)
    return rows


def process_split(split_name: str):
    with open(SPLITS_DIR / f"{split_name}.txt") as f:
        base_names = [line.strip() for line in f if line.strip()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{split_name}_features.csv"

    fieldnames = ["base_name", "uid", "subtype"] + FEATURE_NAMES

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_buildings = 0
        for i, base in enumerate(base_names):
            rows = extract_features_for_tile(base)
            for row in rows:
                writer.writerow(row)
            total_buildings += len(rows)

            if (i + 1) % 50 == 0:
                print(f"  [{split_name}] {i + 1}/{len(base_names)} tiles, {total_buildings} buildings so far")
                f.flush()

    print(f"[{split_name}] Done: {len(base_names)} tiles, {total_buildings} buildings -> {out_path}")


def main():
    import sys as _sys
    splits_to_run = _sys.argv[1:] if len(_sys.argv) > 1 else ["train", "val", "test"]
    for split in splits_to_run:
        process_split(split)


if __name__ == "__main__":
    main()
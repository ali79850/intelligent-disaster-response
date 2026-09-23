"""
Phase 3, Sub-step 2: Generate single-channel segmentation masks from xBD
post-disaster label JSONs, using the verified features.xy pixel-coordinate
polygons.

Class-to-pixel-value mapping (see DECISIONS.md):
  0 = background, 1 = no-damage, 2 = minor-damage, 3 = major-damage,
  4 = destroyed, 5 = un-classified (excluded from loss/metrics downstream
  via an ignore mask, not via a separate file here)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABELS_DIR = Path("data/raw/xbd/train/labels")
OUTPUT_DIR = Path("data/processed/masks")

CLASS_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,
}

IMAGE_SIZE = 1024  # verified uniform in Phase 2


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    coords_str = wkt[wkt.index("((") + 2 : wkt.index("))")]
    return [tuple(map(float, pair.split())) for pair in coords_str.split(",")]


def generate_mask(label_path: Path) -> np.ndarray:
    with open(label_path) as f:
        data = json.load(f)

    mask_img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    draw = ImageDraw.Draw(mask_img)

    for feat in data["features"]["xy"]:
        subtype = feat["properties"].get("subtype", "un-classified")
        pixel_value = CLASS_MAP.get(subtype, CLASS_MAP["un-classified"])
        points = parse_wkt_polygon(feat["wkt"])
        draw.polygon(points, fill=pixel_value)

    return np.array(mask_img)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label_files = sorted(LABELS_DIR.glob("*post_disaster.json"))

    print(f"Generating masks for {len(label_files)} post-disaster labels...")
    class_pixel_totals = {v: 0 for v in range(6)}

    for i, label_path in enumerate(label_files):
        mask = generate_mask(label_path)
        base_name = label_path.name.replace("_post_disaster.json", "")
        out_path = OUTPUT_DIR / f"{base_name}_mask.png"
        Image.fromarray(mask).save(out_path)

        values, counts = np.unique(mask, return_counts=True)
        for v, c in zip(values, counts):
            class_pixel_totals[int(v)] += int(c)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(label_files)} done")

    print(f"\nDone. Masks saved to {OUTPUT_DIR}")
    total_pixels = sum(class_pixel_totals.values())
    print("\n=== Pixel-level class distribution (all masks) ===")
    class_names = {0: "background", 1: "no-damage", 2: "minor-damage",
                   3: "major-damage", 4: "destroyed", 5: "un-classified"}
    for v in range(6):
        pct = 100 * class_pixel_totals[v] / total_pixels
        print(f"  {v} ({class_names[v]:15s}): {class_pixel_totals[v]:12d}  ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
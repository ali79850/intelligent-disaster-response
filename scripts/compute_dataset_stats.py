"""
Phase 3, Sub-step 4a: Compute real per-channel mean/std from the training
split's images, for use in normalization — rather than assuming ImageNet
statistics transfer to satellite imagery.

Computed over TRAIN split only (both pre- and post-disaster images), per
standard practice: never compute normalization stats using val/test data,
that would be a form of data leakage.
"""
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES_DIR = Path("data/raw/xbd/train/images")
SPLITS_DIR = Path("data/interim/splits")

def main():
    with open(SPLITS_DIR / "train.txt") as f:
        base_names = [line.strip() for line in f if line.strip()]

    print(f"Computing stats over {len(base_names)} train tiles (pre + post images)...")

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for i, base in enumerate(base_names):
        for suffix in ["pre_disaster", "post_disaster"]:
            img_path = IMAGES_DIR / f"{base}_{suffix}.png"
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(base_names)} tiles done")

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    print(f"\n=== Dataset statistics (train split, RGB, [0,1] scale) ===")
    print(f"  mean: {mean.tolist()}")
    print(f"  std:  {std.tolist()}")

if __name__ == "__main__":
    main()
"""
Phase 3, Sub-step 2b: Visually verify one generated mask against its source
post-disaster image, to confirm rasterization actually worked correctly
before trusting these masks for training.
"""
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES_DIR = Path("data/raw/xbd/train/images")
MASKS_DIR = Path("data/processed/masks")
OUTPUT_DIR = Path("reports/figures")

COLOR_MAP = {
    0: (0, 0, 0),        # background - black
    1: (0, 255, 0),      # no-damage - green
    2: (255, 255, 0),    # minor-damage - yellow
    3: (255, 128, 0),    # major-damage - orange
    4: (255, 0, 0),      # destroyed - red
    5: (128, 128, 128),  # un-classified - gray
}

def colorize_mask(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return Image.fromarray(rgb)

def main():
    base_name = "hurricane-matthew_00000000"

    post_img = Image.open(IMAGES_DIR / f"{base_name}_post_disaster.png").convert("RGB")
    mask = np.array(Image.open(MASKS_DIR / f"{base_name}_mask.png"))
    mask_colored = colorize_mask(mask)

    # Side-by-side comparison
    combined = Image.new("RGB", (post_img.width * 2, post_img.height))
    combined.paste(post_img, (0, 0))
    combined.paste(mask_colored, (post_img.width, 0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{base_name}_mask_verification.png"
    combined.save(out_path)
    print(f"Saved side-by-side comparison: {out_path}")

    # Also blend mask over image for direct overlap check
    mask_rgba = mask_colored.convert("RGBA")
    mask_rgba.putalpha(120)
    overlay = post_img.convert("RGBA")
    overlay.paste(mask_rgba, (0, 0), mask_rgba)
    overlay_path = OUTPUT_DIR / f"{base_name}_mask_overlay.png"
    overlay.convert("RGB").save(overlay_path)
    print(f"Saved blended overlay: {overlay_path}")

if __name__ == "__main__":
    main()
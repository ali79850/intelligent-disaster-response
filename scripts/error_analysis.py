"""
Phase 6: Error analysis - visualize specific failure/success cases from the
trained U-Net (epoch 10) against ground truth, informed by what the
confusion matrix already told us (Phase 5): destroyed has high recall but
low precision; minor/major-damage are weak across the board.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.xbd_dataset import XBDDataset
from src.models.unet import UNetResNet18

IMAGES_DIR = Path("data/raw/xbd/train/images")
OUTPUT_DIR = Path("reports/figures/error_analysis")

COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 128, 0),
    4: (255, 0, 0),
}

def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb

def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """White = correct, red = wrong, black = ignored (255)."""
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = gt != 255
    correct = valid & (gt == pred)
    wrong = valid & (gt != pred)
    rgb[correct] = (255, 255, 255)
    rgb[wrong] = (255, 0, 0)
    return rgb

def analyze_tile(model, ds, base_name, device):
    idx = ds.base_names.index(base_name)
    sample = ds[idx]

    pre = sample["pre_image"].unsqueeze(0).to(device)
    post = sample["post_image"].unsqueeze(0).to(device)
    gt = sample["mask"].numpy()

    with torch.no_grad():
        output = model(pre, post)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    post_img = Image.open(IMAGES_DIR / f"{base_name}_post_disaster.png").convert("RGB")
    gt_colored = Image.fromarray(colorize(gt))
    pred_colored = Image.fromarray(colorize(pred))
    error_map = Image.fromarray(make_error_map(gt, pred))

    size = post_img.width
    combined = Image.new("RGB", (size * 4, size))
    combined.paste(post_img, (0, 0))
    combined.paste(gt_colored, (size, 0))
    combined.paste(pred_colored, (size * 2, 0))
    combined.paste(error_map, (size * 3, 0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{base_name}_error_analysis.png"
    combined.save(out_path)

    # Print per-class breakdown for this specific tile
    print(f"\n{base_name}:")
    for cls in range(5):
        gt_count = int((gt == cls).sum())
        pred_count = int((pred == cls).sum())
        if gt_count > 0 or pred_count > 0:
            print(f"  class {cls}: GT pixels={gt_count}, predicted pixels={pred_count}")
    print(f"  Saved: {out_path}")

def main():
    device = torch.device("cpu")
    model = UNetResNet18(num_classes=5, pretrained=False).to(device)
    model.load_state_dict(torch.load("models/unet_epoch10.pt", map_location=device))
    model.eval()

    ds = XBDDataset("test", augment=False)

    # Deliberately chosen: known damage-type tiles from test set, per
    # Phase 2/4 findings, not random - we want to SEE the destroyed
    # precision problem and the minor/major weakness directly.
    tiles_to_analyze = [
        "hurricane-matthew_00000000",   # known tile, minor-damage-heavy
        "santa-rosa-wildfire_00000001", # fire event, destroyed-heavy per Phase 2
        "palu-tsunami_00000001",        # tsunami event, destroyed-heavy per Phase 2
    ]

    for base_name in tiles_to_analyze:
        if base_name not in ds.base_names:
            print(f"WARNING: {base_name} not in test set, skipping")
            continue
        analyze_tile(model, ds, base_name, device)

if __name__ == "__main__":
    main()
"""
Phase 7: Generate a targeted Grad-CAM explanation for the specific wrong
"destroyed" prediction identified in Phase 6 error analysis
(hurricane-matthew_00000000: model predicted destroyed where ground truth
says minor-damage).
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.xbd_dataset import XBDDataset
from src.models.unet import UNetResNet18
from src.explainability.grad_cam import SegmentationGradCAM

IMAGES_DIR = Path("data/raw/xbd/train/images")
OUTPUT_DIR = Path("reports/figures/error_analysis")

def overlay_heatmap(image: Image.Image, cam: np.ndarray) -> Image.Image:
    """Red-tinted overlay: brighter red = higher Grad-CAM activation."""
    heatmap = np.zeros((*cam.shape, 3), dtype=np.uint8)
    heatmap[:, :, 0] = (cam * 255).astype(np.uint8)  # red channel only

    heatmap_img = Image.fromarray(heatmap).convert("RGBA")
    heatmap_img.putalpha(140)

    base = image.convert("RGBA")
    return Image.alpha_composite(base, heatmap_img).convert("RGB")

def main():
    device = torch.device("cpu")
    model = UNetResNet18(num_classes=5, pretrained=False).to(device)
    model.load_state_dict(torch.load("models/unet_epoch10.pt", map_location=device))
    model.eval()

    grad_cam = SegmentationGradCAM(model, target_layer=model.enc4)

    base_name = "hurricane-matthew_00000000"
    ds = XBDDataset("test", augment=False)
    idx = ds.base_names.index(base_name)
    sample = ds[idx]

    pre = sample["pre_image"].unsqueeze(0).to(device)
    post = sample["post_image"].unsqueeze(0).to(device)
    gt = sample["mask"]

    with torch.no_grad():
        output = model(pre, post)
        pred = output.argmax(dim=1).squeeze(0)

    # Region of interest: pixels where model predicted destroyed (4) but
    # ground truth says something else (the specific wrong decision)
    wrong_destroyed_region = (pred == 4) & (gt != 4) & (gt != 255)
    print(f"Pixels in region of interest (wrongly predicted destroyed): {wrong_destroyed_region.sum().item()}")

    if wrong_destroyed_region.sum() == 0:
        print("No wrong-destroyed pixels found in this tile - nothing to explain")
        return

    cam = grad_cam.generate(pre, post, target_class=4, region_mask=wrong_destroyed_region)

    post_img = Image.open(IMAGES_DIR / f"{base_name}_post_disaster.png").convert("RGB")
    overlay = overlay_heatmap(post_img, cam)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{base_name}_gradcam_wrong_destroyed.png"
    overlay.save(out_path)
    print(f"Saved: {out_path}")
    print("\nREMINDER: This heatmap shows where the model's internal gradients")
    print("were most sensitive for THIS specific wrong prediction. It does not")
    print("prove causal necessity or capture the full decoder pathway.")

if __name__ == "__main__":
    main()
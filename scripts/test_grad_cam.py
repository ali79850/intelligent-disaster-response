"""
Phase 7: Verify SegmentationGradCAM produces a sane heatmap before using it
for real error-analysis interpretation.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.data.xbd_dataset import XBDDataset
from src.models.unet import UNetResNet18
from src.explainability.grad_cam import SegmentationGradCAM

def main():
    device = torch.device("cpu")
    model = UNetResNet18(num_classes=5, pretrained=False).to(device)
    model.load_state_dict(torch.load("models/unet_epoch10.pt", map_location=device))
    model.eval()

    grad_cam = SegmentationGradCAM(model, target_layer=model.enc4)

    ds = XBDDataset("test", augment=False)
    idx = ds.base_names.index("hurricane-matthew_00000000")
    sample = ds[idx]

    pre = sample["pre_image"].unsqueeze(0).to(device)
    post = sample["post_image"].unsqueeze(0).to(device)

    cam = grad_cam.generate(pre, post, target_class=4)  # destroyed

    print(f"CAM shape: {cam.shape}")
    print(f"CAM value range: [{cam.min():.4f}, {cam.max():.4f}]")
    print(f"CAM mean: {cam.mean():.4f}")
    print("Should be shape (1024, 1024), range [0, 1]")

if __name__ == "__main__":
    main()
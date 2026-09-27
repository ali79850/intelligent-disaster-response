import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from src.data.xbd_dataset import XBDDataset
from src.models.unet import UNetResNet18

device = torch.device("cpu")
model = UNetResNet18(num_classes=5, pretrained=False).to(device)
model.load_state_dict(torch.load("models/unet_epoch10.pt", map_location=device))
model.eval()

ds = XBDDataset("test", augment=False)
idx = ds.base_names.index("santa-rosa-wildfire_00000137")
sample = ds[idx]

pre = sample["pre_image"].unsqueeze(0)
post = sample["post_image"].unsqueeze(0)
gt = sample["mask"].numpy()

with torch.no_grad():
    output = model(pre, post)
    pred = output.argmax(dim=1).squeeze(0).numpy()

gt_destroyed = (gt == 4)
pred_destroyed = (pred == 4)
overlap = gt_destroyed & pred_destroyed

print(f"Ground truth destroyed pixels: {gt_destroyed.sum()}")
print(f"Predicted destroyed pixels: {pred_destroyed.sum()}")
print(f"Overlap (correctly found): {overlap.sum()}")
print(f"GT destroyed pixels the model MISSED entirely: {(gt_destroyed & ~pred_destroyed).sum()}")
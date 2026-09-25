"""
Phase 5: Verify UNetResNet18 produces correctly-shaped output on a real
batch, before attempting any training.
"""
import sys
sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader
from src.data.xbd_dataset import XBDDataset
from src.models.unet import UNetResNet18

def main():
    ds = XBDDataset("train")
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print("Loading UNetResNet18 (pretrained=True) - this will download ResNet-18 weights on first run...")
    model = UNetResNet18(num_classes=5, pretrained=True)
    model.eval()

    with torch.no_grad():
        output = model(batch["pre_image"], batch["post_image"])

    print(f"Model output shape: {tuple(output.shape)}")
    print(f"Expected: (2, 5, 1024, 1024)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")

if __name__ == "__main__":
    main()
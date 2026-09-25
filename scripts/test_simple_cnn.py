"""
Phase 4: Verify SimpleCNN produces correctly-shaped output on a real batch,
before attempting any training.
"""
import sys
sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader
from src.data.xbd_dataset import XBDDataset
from src.models.simple_cnn import SimpleCNN

def main():
    ds = XBDDataset("train")
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = SimpleCNN(num_classes=5)
    model.eval()

    with torch.no_grad():
        output = model(batch["pre_image"], batch["post_image"])

    print(f"Input pre_image shape:  {tuple(batch['pre_image'].shape)}")
    print(f"Input post_image shape: {tuple(batch['post_image'].shape)}")
    print(f"Model output shape:     {tuple(output.shape)}")
    print(f"Expected: (2, 5, 1024, 1024)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")

if __name__ == "__main__":
    main()
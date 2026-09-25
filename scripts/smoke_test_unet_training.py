"""
Phase 5: Smoke test - train the real UNetResNet18 on a tiny subset for a
few iterations, confirm loss decreases, AND measure timing per iteration.
Given the U-Net's much larger size (14.3M params vs Baseline 3's 91k),
this early timing signal tells us honestly whether local CPU training is
feasible for a full run, or whether we should move to Colab immediately.
"""
import sys
sys.path.insert(0, ".")

import time
import torch
from torch.utils.data import DataLoader, Subset

from src.data.xbd_dataset import XBDDataset
from src.models.unet import UNetResNet18
from src.training.losses import CombinedLoss

CLASS_WEIGHTS = torch.tensor([
    1.0 / 0.9412, 1.0 / 0.0430, 1.0 / 0.0053, 1.0 / 0.0068, 1.0 / 0.0032,
])
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum() * 5

def main():
    ds = XBDDataset("train")
    small_subset = Subset(ds, list(range(5)))
    loader = DataLoader(small_subset, batch_size=1, shuffle=True, num_workers=0)

    print("Loading UNetResNet18...")
    model = UNetResNet18(num_classes=5, pretrained=True)
    criterion = CombinedLoss(class_weights=CLASS_WEIGHTS, num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    print("Smoke test: training on 5 tiles for 5 iterations (timing each)...")

    for iteration in range(5):
        iter_start = time.time()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch["pre_image"], batch["post_image"])
            loss = criterion(output, batch["mask"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        iter_time = time.time() - iter_start

        avg_loss = total_loss / len(loader)
        print(f"  Iteration {iteration + 1}/5: avg loss = {avg_loss:.4f}, time = {iter_time:.1f}s "
              f"({iter_time/len(loader):.1f}s/sample)")

    print("\nSmoke test complete.")

if __name__ == "__main__":
    main()
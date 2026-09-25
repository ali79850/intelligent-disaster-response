"""
Phase 4: Smoke test - train SimpleCNN on a tiny subset for a few iterations,
confirm loss decreases, BEFORE committing to any full training run.
Lesson from the feature-extraction hang: verify on a small scale first.
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.xbd_dataset import XBDDataset
from src.models.simple_cnn import SimpleCNN

# Inverse-frequency class weights from Phase 3 pixel-level distribution
# background=0.9412, no-damage=0.0430, minor=0.0053, major=0.0068, destroyed=0.0032
CLASS_WEIGHTS = torch.tensor([
    1.0 / 0.9412,   # 0 background
    1.0 / 0.0430,   # 1 no-damage
    1.0 / 0.0053,   # 2 minor-damage
    1.0 / 0.0068,   # 3 major-damage
    1.0 / 0.0032,   # 4 destroyed
])
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum() * 5  # normalize, keep scale reasonable

def main():
    ds = XBDDataset("train")
    small_subset = Subset(ds, list(range(5)))  # just 5 tiles
    loader = DataLoader(small_subset, batch_size=1, shuffle=True, num_workers=0)

    model = SimpleCNN(num_classes=5)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    print("Smoke test: training on 5 tiles for 10 iterations...")
    for iteration in range(10):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch["pre_image"], batch["post_image"])
            loss = criterion(output, batch["mask"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Iteration {iteration + 1}/10: avg loss = {avg_loss:.4f}")

    print("\nSmoke test complete. Loss should show a general downward trend.")

if __name__ == "__main__":
    main()
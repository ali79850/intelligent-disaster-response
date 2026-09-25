"""
Phase 4, Baseline 3: Full training run for SimpleCNN. Times the first
epoch explicitly to give an honest, measured basis for deciding how many
epochs are feasible on this CPU-only setup - not a guess.
"""
import sys
sys.path.insert(0, ".")

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.xbd_dataset import XBDDataset
from src.models.simple_cnn import SimpleCNN

CLASS_WEIGHTS = torch.tensor([
    1.0 / 0.9412, 1.0 / 0.0430, 1.0 / 0.0053, 1.0 / 0.0068, 1.0 / 0.0032,
])
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum() * 5

NUM_EPOCHS = 3  # deliberately small, per our Phase 4 compute plan

def main():
    train_ds = XBDDataset("train")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

    model = SimpleCNN(num_classes=5)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training for up to {NUM_EPOCHS} epochs, {len(train_loader)} batches/epoch...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(batch["pre_image"], batch["post_image"])
            loss = criterion(output, batch["mask"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - epoch_start
                print(f"  Epoch {epoch+1}, batch {i+1}/{len(train_loader)}, "
                      f"avg loss so far: {total_loss/(i+1):.4f}, elapsed: {elapsed:.1f}s")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} DONE: avg loss = {avg_loss:.4f}, time = {epoch_time:.1f}s ({epoch_time/60:.1f} min)")

        torch.save(model.state_dict(), f"models/baseline3_epoch{epoch+1}.pt")
        print(f"  Saved checkpoint: models/baseline3_epoch{epoch+1}.pt")

if __name__ == "__main__":
    main()
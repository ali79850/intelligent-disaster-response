"""
Phase 3, final step: Verify DataLoader batches correctly - correct batch
dimension, correct shapes, and confirm we can iterate through at least
one full batch without error before considering Phase 3 complete.
"""
import sys
sys.path.insert(0, ".")

import time
from torch.utils.data import DataLoader
from src.data.xbd_dataset import XBDDataset

def main():
    train_ds = XBDDataset("train")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

    print(f"Train dataset: {len(train_ds)} samples, {len(train_loader)} batches at batch_size=2")

    start = time.time()
    batch = next(iter(train_loader))
    elapsed = time.time() - start

    print(f"\nFirst batch loaded in {elapsed:.2f}s")
    print(f"  pre_image  shape: {tuple(batch['pre_image'].shape)}")
    print(f"  post_image shape: {tuple(batch['post_image'].shape)}")
    print(f"  mask       shape: {tuple(batch['mask'].shape)}")
    print(f"  base_names: {batch['base_name']}")

    # Time a few more batches to get a rough per-batch loading estimate
    print("\nTiming 5 batches for rough throughput estimate...")
    start = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 4:
            break
    elapsed = time.time() - start
    print(f"5 batches in {elapsed:.2f}s ({elapsed/5:.2f}s/batch average)")

if __name__ == "__main__":
    main()
"""
Phase 4, Baseline 3: Evaluate the trained SimpleCNN on val and test sets.

Accumulates a running confusion matrix per batch rather than storing all
raw pixels in memory (the original version crashed trying to hold ~461
million pixels for val alone - see DECISIONS.md).
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.xbd_dataset import XBDDataset
from src.models.simple_cnn import SimpleCNN

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES = 5

def evaluate(model, loader, split_name):
    model.eval()
    # Running confusion matrix: rows=true, cols=pred
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            output = model(batch["pre_image"], batch["post_image"])
            pred = output.argmax(dim=1)
            mask = batch["mask"]

            valid = mask != 255
            p = pred[valid].numpy()
            t = mask[valid].numpy()

            # Accumulate into confusion matrix without storing raw pixels
            for true_c in range(NUM_CLASSES):
                for pred_c in range(NUM_CLASSES):
                    conf_matrix[true_c, pred_c] += int(np.sum((t == true_c) & (p == pred_c)))

            if (i + 1) % 50 == 0:
                print(f"  [{split_name}] {i + 1}/{len(loader)} batches evaluated")

    print(f"\n=== {split_name.upper()} SET RESULTS ===")
    print("Confusion matrix (rows=true, cols=pred):")
    print("Classes:", CLASS_NAMES)
    print(conf_matrix)

    print(f"\n{'Class':15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    f1_scores = []
    for c in range(NUM_CLASSES):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        support = conf_matrix[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        print(f"{CLASS_NAMES[c]:15s} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}")

    macro_f1 = np.mean(f1_scores)
    print(f"\nMacro F1: {macro_f1:.4f}")

def main():
    model = SimpleCNN(num_classes=5)
    model.load_state_dict(torch.load("models/baseline3_epoch3.pt"))

    val_ds = XBDDataset("val", augment=False)
    test_ds = XBDDataset("test", augment=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    evaluate(model, val_loader, "val")
    evaluate(model, test_loader, "test")

if __name__ == "__main__":
    main()
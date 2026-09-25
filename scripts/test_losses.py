"""
Phase 5: Verify CombinedLoss computes sensible values and handles
ignore_index correctly, before using it in any training loop.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.training.losses import CombinedLoss

CLASS_WEIGHTS = torch.tensor([
    1.0 / 0.9412, 1.0 / 0.0430, 1.0 / 0.0053, 1.0 / 0.0068, 1.0 / 0.0032,
])
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum() * 5

def main():
    torch.manual_seed(42)
    B, C, H, W = 2, 5, 8, 8

    logits = torch.randn(B, C, H, W, requires_grad=True)
    targets = torch.randint(0, 5, (B, H, W))
    # Inject some ignore_index pixels to confirm they're handled
    targets[0, 0, 0] = 255
    targets[1, 3, 3] = 255

    criterion = CombinedLoss(class_weights=CLASS_WEIGHTS, num_classes=5)
    loss = criterion(logits, targets)

    print(f"Combined loss value: {loss.item():.4f}")
    print("Should be a positive, finite number")

    loss.backward()
    print(f"\nBackward pass succeeded. Gradient exists on logits: {logits.grad is not None}")
    print(f"Gradient contains NaN: {torch.isnan(logits.grad).any().item()}")

    # Sanity check: perfect predictions should give near-zero loss
    perfect_logits = torch.zeros(B, C, H, W)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                if targets[b, h, w] != 255:
                    perfect_logits[b, targets[b, h, w], h, w] = 10.0
    perfect_loss = criterion(perfect_logits, targets)
    print(f"\nLoss on near-perfect predictions: {perfect_loss.item():.4f} (should be much lower than random)")

if __name__ == "__main__":
    main()
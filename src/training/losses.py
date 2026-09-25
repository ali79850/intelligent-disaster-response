"""
Phase 5: Combined weighted CrossEntropy + Dice loss for segmentation.

Dice loss directly optimizes region overlap, which literature and our own
Baseline 3 finding both suggest helps small/rare classes get more useful
gradient signal than cross-entropy alone (which let Baseline 3 collapse
to predicting only background/no-damage despite already using class
weighting there).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W) raw model output
        targets: (B, H, W) class indices, may contain IGNORE_INDEX
        """
        valid_mask = (targets != IGNORE_INDEX)
        # Replace ignored pixels with 0 temporarily so one_hot doesn't error;
        # they're excluded from the loss sum via valid_mask below regardless.
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_onehot = F.one_hot(safe_targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask_expanded
        targets_onehot = targets_onehot * valid_mask_expanded

        dims = (0, 2, 3)  # sum over batch, height, width - per-class dice
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, num_classes: int = 5, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + self.dice_weight * dice_loss
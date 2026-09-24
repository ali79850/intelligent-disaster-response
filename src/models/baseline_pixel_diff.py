"""
Phase 4, Baseline 1: Naive pixel-difference change detection.

Computes absolute pixel-wise difference between pre- and post-disaster
images, thresholds it to produce a binary changed/unchanged prediction.
This baseline cannot distinguish damage severity - it is evaluated only
as a binary "any damage vs no damage" detector, which is an honest framing
of what this method can actually do.
"""
import numpy as np
from PIL import Image


def compute_pixel_diff_prediction(pre_path: str, post_path: str, threshold: float = 30.0) -> np.ndarray:
    """
    Returns a binary (0/1) prediction mask: 1 = predicted changed/damaged.
    threshold is on a 0-255 grayscale difference scale.
    """
    pre = np.array(Image.open(pre_path).convert("L"), dtype=np.float32)
    post = np.array(Image.open(post_path).convert("L"), dtype=np.float32)
    diff = np.abs(post - pre)
    return (diff > threshold).astype(np.uint8)


def load_binary_ground_truth(mask_path: str) -> np.ndarray:
    """
    Collapses the 6-value mask (0=background, 1=no-damage, 2-4=damage
    severity, 5=un-classified) into binary: 0 = no-damage-or-background,
    1 = any real damage. Un-classified pixels are excluded (set to -1,
    filtered out during evaluation) since we have no reliable ground truth
    for them, consistent with the Phase 3 ignore-index decision.
    """
    mask = np.array(Image.open(mask_path))
    binary = np.zeros_like(mask, dtype=np.int8)
    binary[np.isin(mask, [2, 3, 4])] = 1   # minor/major/destroyed = damage
    binary[mask == 5] = -1                  # un-classified = ignore
    # 0 (background) and 1 (no-damage) both remain 0 = "no damage"
    return binary
"""
Phase 4: Reusable binary segmentation evaluation metrics (IoU, F1, precision,
recall), with support for an ignore mask (e.g., un-classified pixels).

Used across all baselines and the eventual real model, so metric
computation is identical and comparisons are meaningful.
"""
import numpy as np


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    pred: binary array (0/1)
    gt: array with 0, 1, and optionally -1 for ignored pixels
    Ignored pixels are excluded entirely from all counts.
    """
    valid = gt != -1
    pred_v = pred[valid]
    gt_v = gt[valid]

    tp = int(np.sum((pred_v == 1) & (gt_v == 1)))
    fp = int(np.sum((pred_v == 1) & (gt_v == 0)))
    fn = int(np.sum((pred_v == 0) & (gt_v == 1)))
    tn = int(np.sum((pred_v == 0) & (gt_v == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "iou": iou,
    }
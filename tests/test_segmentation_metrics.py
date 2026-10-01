"""
Phase 12: Tests for binary segmentation metrics. Uses hand-constructed
arrays with known, hand-calculated expected results - not placeholder
assertions.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.evaluation.segmentation_metrics import compute_binary_metrics


def test_perfect_prediction():
    pred = np.array([1, 1, 0, 0])
    gt = np.array([1, 1, 0, 0])
    m = compute_binary_metrics(pred, gt)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["iou"] == 1.0


def test_known_partial_overlap():
    # 2 true positives, 1 false positive, 1 false negative, 0 true negatives
    pred = np.array([1, 1, 1, 0])
    gt = np.array([1, 1, 0, 1])
    m = compute_binary_metrics(pred, gt)
    # precision = tp/(tp+fp) = 2/3
    assert abs(m["precision"] - (2 / 3)) < 1e-6
    # recall = tp/(tp+fn) = 2/3
    assert abs(m["recall"] - (2 / 3)) < 1e-6
    # iou = tp/(tp+fp+fn) = 2/4 = 0.5
    assert abs(m["iou"] - 0.5) < 1e-6


def test_ignore_pixels_excluded():
    pred = np.array([1, 1, 0, 0])
    gt = np.array([1, -1, 0, 0])  # one ignored pixel
    m = compute_binary_metrics(pred, gt)
    # Only 3 valid pixels should be counted; the ignored pixel must not
    # count as a false positive even though pred=1 there
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 3


def test_no_positive_predictions_or_ground_truth():
    pred = np.array([0, 0, 0])
    gt = np.array([0, 0, 0])
    m = compute_binary_metrics(pred, gt)
    # Should not divide by zero; precision/recall/f1/iou default to 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["iou"] == 0.0
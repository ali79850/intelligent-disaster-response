"""
Phase 12: Tests for report generation logic, using hand-constructed fake
inference results - no model loading required, since this tests pure
aggregation/calculation logic, not the model itself.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.inference.report import generate_report, LOW_CONFIDENCE_THRESHOLD


def make_fake_inference_result(pred_mask, probs, model_version="test_model"):
    return {
        "model_version": model_version,
        "predicted_mask": pred_mask,
        "class_probabilities": probs,
        "class_statistics": {
            name: {
                "pixel_count": int((pred_mask == i).sum()),
                "percentage": round(100 * (pred_mask == i).sum() / pred_mask.size, 2),
            }
            for i, name in enumerate(["background", "no-damage", "minor-damage", "major-damage", "destroyed"])
        },
        "mean_confidence": float(probs.max(axis=0).mean()),
        "image_size": pred_mask.shape,
    }


def test_requires_review_triggers_on_low_confidence():
    # 2x2 image: all pixels predicted class 1 (no-damage) with LOW confidence
    pred_mask = np.ones((2, 2), dtype=int)
    probs = np.zeros((5, 2, 2))
    probs[1, :, :] = 0.3  # below LOW_CONFIDENCE_THRESHOLD (0.5)
    result = make_fake_inference_result(pred_mask, probs)

    report = generate_report(result, tile_id="test_tile")

    assert report["quality_flags"]["requires_human_review"] is True
    assert "no-damage" in report["quality_flags"]["low_confidence_classes"]
    assert "background" not in report["quality_flags"]["low_confidence_classes"]


def test_no_review_needed_when_confident():
    pred_mask = np.ones((2, 2), dtype=int)
    probs = np.zeros((5, 2, 2))
    probs[1, :, :] = 0.95  # above threshold
    result = make_fake_inference_result(pred_mask, probs)

    report = generate_report(result, tile_id="test_tile")

    assert report["quality_flags"]["requires_human_review"] is False
    assert report["quality_flags"]["low_confidence_classes"] == []


def test_unpredicted_class_has_null_confidence():
    # No pixels predicted as class 2 (minor-damage) at all
    pred_mask = np.zeros((2, 2), dtype=int)
    probs = np.zeros((5, 2, 2))
    probs[0, :, :] = 0.9
    result = make_fake_inference_result(pred_mask, probs)

    report = generate_report(result, tile_id="test_tile")

    assert report["confidence"]["per_class"]["minor-damage"] is None


def test_affected_area_percentage_calculation():
    # 4 pixels total, 1 destroyed = 25% affected
    pred_mask = np.array([[4, 0], [0, 0]])
    probs = np.zeros((5, 2, 2))
    probs[0] = 0.9
    probs[4, 0, 0] = 0.9
    result = make_fake_inference_result(pred_mask, probs)

    report = generate_report(result, tile_id="test_tile")

    assert report["summary"]["affected_area_percentage"] == 25.0


def test_limitations_always_present_and_nonempty():
    pred_mask = np.zeros((2, 2), dtype=int)
    probs = np.zeros((5, 2, 2))
    probs[0] = 0.9
    result = make_fake_inference_result(pred_mask, probs)

    report = generate_report(result, tile_id="test_tile")

    assert len(report["limitations"]) > 0
    assert any("preliminary" in lim.lower() for lim in report["limitations"])
    
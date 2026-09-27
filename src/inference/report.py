"""
Phase 9: Structured analytical report generation from inference engine
output. Per project scope: no LLM-generated numbers here - every value is
computed directly from the model's actual output, nothing invented.
"""
from datetime import datetime, timezone

import numpy as np

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]

# Below this confidence, flag the tile for human review - per original
# spec section 12/13 (uncertainty, human-in-the-loop review)
LOW_CONFIDENCE_THRESHOLD = 0.5


def generate_report(inference_result: dict, tile_id: str) -> dict:
    """
    Builds a structured report dict from a single DamageInferenceEngine
    prediction. This is what an API endpoint or dashboard would consume -
    plain data, no narrative text generation here.
    """
    stats = inference_result["class_statistics"]
    probs = inference_result["class_probabilities"]  # (5, H, W)
    pred_mask = inference_result["predicted_mask"]     # (H, W)

    # Per-class confidence: mean max-probability specifically for pixels
    # predicted as that class (not overall mean, per Phase 9's finding
    # that aggregate confidence can mask class-specific miscalibration)
    per_class_confidence = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask_for_class = (pred_mask == cls_idx)
        if mask_for_class.sum() > 0:
            class_probs_here = probs[cls_idx][mask_for_class]
            per_class_confidence[cls_name] = round(float(class_probs_here.mean()), 4)
        else:
            per_class_confidence[cls_name] = None  # class not predicted anywhere in this tile

    # Flag: does this tile have any damage class prediction with low
    # confidence? A real, computed flag - not a placeholder.
    low_confidence_classes = [
        cls_name for cls_name, conf in per_class_confidence.items()
        if conf is not None and conf < LOW_CONFIDENCE_THRESHOLD and cls_name != "background"
    ]
    requires_review = len(low_confidence_classes) > 0

    total_damage_pixels = sum(
        stats[cls]["pixel_count"] for cls in ["no-damage", "minor-damage", "major-damage", "destroyed"]
    )
    total_pixels = inference_result["image_size"][0] * inference_result["image_size"][1]
    affected_percentage = round(100 * total_damage_pixels / total_pixels, 2)

    return {
        "tile_id": tile_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": inference_result["model_version"],
        "summary": {
            "total_pixels_analyzed": total_pixels,
            "affected_area_percentage": affected_percentage,
        },
        "damage_distribution": stats,
        "confidence": {
            "overall_mean": inference_result["mean_confidence"],
            "per_class": per_class_confidence,
        },
        "quality_flags": {
            "requires_human_review": requires_review,
            "low_confidence_classes": low_confidence_classes,
        },
        "limitations": [
            "This is an AI-generated preliminary damage assessment, not an "
            "authoritative or official determination.",
            "Model shows a documented tendency toward severity miscalibration "
            "(see project error analysis): correctly localizes damage but may "
            "over-predict severe categories.",
            "Un-classified/ambiguous regions in training data were excluded "
            "from model training and are not distinguished in this output.",
            "Model trained and evaluated on the xBD dataset; performance on "
            "imagery from different sensors, resolutions, or geographic "
            "regions not represented in training data is not verified.",
        ],
    }
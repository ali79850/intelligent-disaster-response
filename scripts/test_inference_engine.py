"""
Phase 9: Verify DamageInferenceEngine produces consistent results, using
hurricane-matthew_00000000 - a tile we've already thoroughly analyzed in
Phase 6/7, so we can directly compare this engine's output against known,
already-verified results.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from PIL import Image

from src.inference.engine import DamageInferenceEngine

def main():
    engine = DamageInferenceEngine(checkpoint_path="models/unet_epoch10.pt")
    print(f"Loaded model version: {engine.model_version}")
    print(f"Device: {engine.device}")

    base_name = "hurricane-matthew_00000000"
    images_dir = Path("data/raw/xbd/train/images")
    pre_img = Image.open(images_dir / f"{base_name}_pre_disaster.png")
    post_img = Image.open(images_dir / f"{base_name}_post_disaster.png")

    result = engine.predict(pre_img, post_img)

    print(f"\nPredicted mask shape: {result['predicted_mask'].shape}")
    print(f"Class probabilities shape: {result['class_probabilities'].shape}")
    print(f"Mean confidence: {result['mean_confidence']}")
    print(f"\nClass statistics:")
    for cls_name, stats in result["class_statistics"].items():
        print(f"  {cls_name:15s} {stats['pixel_count']:8d} pixels ({stats['percentage']}%)")

    # Sanity check against Phase 6's known result for this exact tile:
    # class 4 (destroyed) predicted pixels should be ~61130
    print(f"\nExpected from Phase 6 error analysis: destroyed ~61130 pixels")
    print(f"This engine reports: {result['class_statistics']['destroyed']['pixel_count']} pixels")

if __name__ == "__main__":
    main()
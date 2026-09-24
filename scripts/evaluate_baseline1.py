"""
Phase 4, Baseline 1: Full evaluation on the test split.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.models.baseline_pixel_diff import compute_pixel_diff_prediction, load_binary_ground_truth
from src.evaluation.segmentation_metrics import compute_binary_metrics

def main():
    with open("data/interim/splits/test.txt") as f:
        base_names = [line.strip() for line in f if line.strip()]

    images_dir = Path("data/raw/xbd/train/images")
    masks_dir = Path("data/processed/masks")

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    print(f"Evaluating Baseline 1 on {len(base_names)} test tiles...")
    for i, base in enumerate(base_names):
        pre_path = images_dir / f"{base}_pre_disaster.png"
        post_path = images_dir / f"{base}_post_disaster.png"
        mask_path = masks_dir / f"{base}_mask.png"

        pred = compute_pixel_diff_prediction(str(pre_path), str(post_path))
        gt = load_binary_ground_truth(str(mask_path))
        m = compute_binary_metrics(pred, gt)

        for k in totals:
            totals[k] += m[k]

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(base_names)} done")

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) > 0 else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = totals["tp"] / (totals["tp"] + totals["fp"] + totals["fn"]) if (totals["tp"] + totals["fp"] + totals["fn"]) > 0 else 0.0

    print(f"\n=== Baseline 1 (naive pixel-diff) - TEST SET RESULTS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    print(f"\nRaw counts: TP={totals['tp']}, FP={totals['fp']}, FN={totals['fn']}, TN={totals['tn']}")

if __name__ == "__main__":
    main()
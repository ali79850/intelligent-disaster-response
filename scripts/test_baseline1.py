"""
Phase 4: Quick sanity check of Baseline 1 on a known tile before running
full evaluation across the test set.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.models.baseline_pixel_diff import compute_pixel_diff_prediction, load_binary_ground_truth

def main():
    base = "hurricane-matthew_00000000"
    pre_path = f"data/raw/xbd/train/images/{base}_pre_disaster.png"
    post_path = f"data/raw/xbd/train/images/{base}_post_disaster.png"
    mask_path = f"data/processed/masks/{base}_mask.png"

    pred = compute_pixel_diff_prediction(pre_path, post_path)
    gt = load_binary_ground_truth(mask_path)

    print(f"Prediction shape: {pred.shape}, unique values: {np.unique(pred)}")
    print(f"Ground truth shape: {gt.shape}, unique values: {np.unique(gt)}")
    print(f"Predicted 'changed' pixels: {pred.sum()} ({100*pred.mean():.2f}% of tile)")
    print(f"Ground truth 'damage' pixels: {(gt==1).sum()} ({100*(gt==1).mean():.2f}% of tile)")
    print(f"Ignored (un-classified) pixels: {(gt==-1).sum()}")

if __name__ == "__main__":
    main()
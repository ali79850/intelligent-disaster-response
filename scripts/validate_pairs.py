"""
Phase 3, Sub-step 3: Verify every tile referenced in each split file has a
complete set of: pre-image, post-image, post-disaster label JSON, and
generated mask. Catch silent mismatches before building the DataLoader.
"""
from pathlib import Path

IMAGES_DIR = Path("data/raw/xbd/train/images")
LABELS_DIR = Path("data/raw/xbd/train/labels")
MASKS_DIR = Path("data/processed/masks")
SPLITS_DIR = Path("data/interim/splits")

def main():
    total_issues = 0

    for split_name in ["train", "val", "test"]:
        split_file = SPLITS_DIR / f"{split_name}.txt"
        with open(split_file) as f:
            base_names = [line.strip() for line in f if line.strip()]

        print(f"\n=== Validating {split_name} ({len(base_names)} entries) ===")
        missing = {"pre_image": [], "post_image": [], "post_label": [], "mask": []}

        for base in base_names:
            if not (IMAGES_DIR / f"{base}_pre_disaster.png").exists():
                missing["pre_image"].append(base)
            if not (IMAGES_DIR / f"{base}_post_disaster.png").exists():
                missing["post_image"].append(base)
            if not (LABELS_DIR / f"{base}_post_disaster.json").exists():
                missing["post_label"].append(base)
            if not (MASKS_DIR / f"{base}_mask.png").exists():
                missing["mask"].append(base)

        split_issues = sum(len(v) for v in missing.values())
        total_issues += split_issues

        if split_issues == 0:
            print(f"  All {len(base_names)} entries complete (pre, post, label, mask).")
        else:
            for kind, items in missing.items():
                if items:
                    print(f"  MISSING {kind}: {len(items)} -> {items[:5]}{'...' if len(items) > 5 else ''}")

    print(f"\n{'='*50}")
    if total_issues == 0:
        print("VALIDATION PASSED: no missing files across all splits.")
    else:
        print(f"VALIDATION FAILED: {total_issues} total missing files found.")

if __name__ == "__main__":
    main()
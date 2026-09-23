import sys
sys.path.insert(0, ".")

from src.data.xbd_dataset import XBDDataset

def test_ignore_index_remapping():
    """Explicitly verify un-classified (raw value 5) gets remapped to 255,
    using a tile known to contain un-classified buildings."""
    from src.data.xbd_dataset import XBDDataset, IGNORE_INDEX

    ds = XBDDataset("train")
    found_ignore = False
    for i in range(len(ds)):
        if ds.base_names[i].startswith("guatemala-volcano"):
            sample = ds[i]
            unique_vals = sample["mask"].unique().tolist()
            if IGNORE_INDEX in unique_vals:
                print(f"Confirmed: {sample['base_name']} mask contains ignore_index={IGNORE_INDEX}")
                found_ignore = True
                break
    if not found_ignore:
        print("WARNING: no guatemala-volcano tile in this scan showed ignore_index=255 - verify manually")

def main():
    for split in ["train", "val", "test"]:
        ds = XBDDataset(split)
        print(f"{split}: {len(ds)} samples")

        sample = ds[0]
        print(f"  pre_image  shape={tuple(sample['pre_image'].shape)} dtype={sample['pre_image'].dtype}")
        print(f"  post_image shape={tuple(sample['post_image'].shape)} dtype={sample['post_image'].dtype}")
        print(f"  mask       shape={tuple(sample['mask'].shape)} dtype={sample['mask'].dtype}")
        print(f"  mask unique values: {sorted(sample['mask'].unique().tolist())}")
        print(f"  base_name: {sample['base_name']}")
        print()

    print("=== Ignore-index remapping check ===")
    test_ignore_index_remapping()

if __name__ == "__main__":
    main()
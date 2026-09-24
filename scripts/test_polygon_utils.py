"""
Phase 4: Verify polygon utility correctness against a known tile.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.preprocessing.polygon_utils import get_buildings_for_tile

def main():
    label_path = Path("data/raw/xbd/train/labels/hurricane-matthew_00000000_post_disaster.json")
    buildings = get_buildings_for_tile(label_path)

    print(f"Found {len(buildings)} buildings in {label_path.name}")
    for b in buildings[:5]:
        pixel_count = b["polygon_mask"].sum()
        print(f"  uid={b['uid'][:8]}... subtype={b['subtype']:15s} polygon_pixels={pixel_count}")

if __name__ == "__main__":
    main()
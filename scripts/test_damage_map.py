"""
Phase 8: Generate a real damage map for hurricane-matthew_00000000, a
tile we've extensively analyzed already (Phase 3, 6, 7) - so we can
sanity-check this map against known findings (predominantly minor-damage,
a couple destroyed instances).
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.geospatial.damage_map import build_damage_map

def main():
    label_path = Path("data/raw/xbd/train/labels/hurricane-matthew_00000000_post_disaster.json")
    output_path = Path("reports/figures/geospatial/hurricane-matthew_00000000_map.html")
    build_damage_map(label_path, output_path)

if __name__ == "__main__":
    main()
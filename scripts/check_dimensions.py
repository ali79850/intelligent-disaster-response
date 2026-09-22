"""
Phase 2, Step 4: Verify image dimensions and ground sample distance (gsd)
consistency across all label metadata, before assuming a fixed tile size
for Phase 3 preprocessing.
"""
import json
from collections import Counter
from pathlib import Path

LABELS_DIR = Path("data/raw/xbd/train/labels")

def main():
    dimensions = Counter()
    gsd_values = []
    disaster_types = Counter()

    for f in LABELS_DIR.glob("*.json"):
        with open(f, "r") as fh:
            d = json.load(fh)
        meta = d["metadata"]
        dimensions[(meta["width"], meta["height"])] += 1
        gsd_values.append(meta["gsd"])
        disaster_types[meta["disaster_type"]] += 1

    print("=== Image dimensions found ===")
    for dim, count in dimensions.most_common():
        print(f"  {dim[0]}x{dim[1]}: {count} files")

    print(f"\n=== Ground sample distance (gsd) ===")
    print(f"  min: {min(gsd_values):.4f} m/pixel")
    print(f"  max: {max(gsd_values):.4f} m/pixel")
    print(f"  avg: {sum(gsd_values)/len(gsd_values):.4f} m/pixel")

    print(f"\n=== Disaster types ===")
    for dtype, count in disaster_types.most_common():
        print(f"  {dtype:15s} {count}")

if __name__ == "__main__":
    main()
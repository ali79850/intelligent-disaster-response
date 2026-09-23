"""
Phase 3, Sub-step 1b: Generate train/val/test file lists based on the approved
event-based split, and report real per-split damage class distribution to
verify the split is actually usable (not just tile counts).
"""
import json
from collections import Counter
from pathlib import Path

LABELS_DIR = Path("data/raw/xbd/train/labels")
OUTPUT_DIR = Path("data/interim/splits")

SPLIT_EVENTS = {
    "train": ["socal-fire", "hurricane-michael", "hurricane-florence",
              "midwest-flooding", "guatemala-volcano"],
    "val": ["hurricane-harvey", "mexico-earthquake"],
    "test": ["hurricane-matthew", "santa-rosa-wildfire", "palu-tsunami"],
}

def main():
    # Map each post-disaster label file's base name to its event
    base_name_to_event = {}
    for f in LABELS_DIR.glob("*post_disaster.json"):
        with open(f, "r") as fh:
            d = json.load(fh)
        base = f.name.replace("_post_disaster.json", "")
        base_name_to_event[base] = d["metadata"]["disaster"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_class_counts = {}

    for split_name, events in SPLIT_EVENTS.items():
        members = [b for b, ev in base_name_to_event.items() if ev in events]
        members.sort()

        out_file = OUTPUT_DIR / f"{split_name}.txt"
        with open(out_file, "w") as fh:
            fh.write("\n".join(members) + "\n")

        # Real class distribution for this split
        class_counts = Counter()
        for base in members:
            label_path = LABELS_DIR / f"{base}_post_disaster.json"
            with open(label_path) as fh:
                d = json.load(fh)
            for feat in d["features"]["lng_lat"]:
                class_counts[feat["properties"].get("subtype", "MISSING")] += 1

        split_class_counts[split_name] = class_counts
        total_buildings = sum(class_counts.values())
        print(f"\n=== {split_name.upper()} ({len(members)} tiles, {out_file}) ===")
        for cls, count in class_counts.most_common():
            pct = 100 * count / total_buildings
            print(f"  {cls:15s} {count:7d}  ({pct:5.2f}%)")

if __name__ == "__main__":
    main()
"""
Phase 2, Step 2: Real damage class distribution across all post-disaster labels,
plus per-disaster-event breakdown.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS_DIR = Path("data/raw/xbd/train/labels")

def main():
    subtype_counts = Counter()
    event_counts = Counter()
    event_subtype_counts = defaultdict(Counter)
    total_buildings = 0
    tile_count = 0

    for f in sorted(LABELS_DIR.glob("*post_disaster.json")):
        tile_count += 1
        with open(f, "r") as fh:
            d = json.load(fh)

        event = d["metadata"]["disaster"]
        event_counts[event] += 1

        for feat in d["features"]["lng_lat"]:
            subtype = feat["properties"].get("subtype", "MISSING")
            subtype_counts[subtype] += 1
            event_subtype_counts[event][subtype] += 1
            total_buildings += 1

    print(f"Total post-disaster tiles: {tile_count}")
    print(f"Total building instances: {total_buildings}\n")

    print("=== Damage class distribution (all events combined) ===")
    for cls, count in subtype_counts.most_common():
        pct = 100 * count / total_buildings
        print(f"  {cls:15s} {count:7d}  ({pct:5.2f}%)")

    print("\n=== Tiles per disaster event ===")
    for event, count in event_counts.most_common():
        print(f"  {event:25s} {count:5d} tiles")

    print("\n=== Damage distribution per event ===")
    for event in sorted(event_subtype_counts.keys()):
        print(f"\n  {event}:")
        ev_total = sum(event_subtype_counts[event].values())
        for cls, count in event_subtype_counts[event].most_common():
            pct = 100 * count / ev_total
            print(f"    {cls:15s} {count:6d}  ({pct:5.2f}%)")

if __name__ == "__main__":
    main()
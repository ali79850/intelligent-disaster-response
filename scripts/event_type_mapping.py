"""
Phase 3, Sub-step 1a: Confirm which disaster_type each named event belongs to,
and re-surface tile counts per event, to inform the train/val/test split design.
"""
import json
from collections import defaultdict
from pathlib import Path

LABELS_DIR = Path("data/raw/xbd/train/labels")

def main():
    event_type = {}
    event_tile_count = defaultdict(int)

    for f in LABELS_DIR.glob("*post_disaster.json"):
        with open(f, "r") as fh:
            d = json.load(fh)
        meta = d["metadata"]
        event = meta["disaster"]
        event_type[event] = meta["disaster_type"]
        event_tile_count[event] += 1

    print(f"{'Event':25s} {'Disaster Type':15s} {'Tiles':>6s}")
    print("-" * 50)
    for event in sorted(event_tile_count, key=lambda e: -event_tile_count[e]):
        print(f"{event:25s} {event_type[event]:15s} {event_tile_count[event]:6d}")

if __name__ == "__main__":
    main()
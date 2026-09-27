"""
Phase 8: Verify what geospatial information is actually available in xBD
metadata and building polygons, before designing any geospatial feature.
Do not assume - check the real, full metadata and polygon coordinate
ranges directly.
"""
import json
from pathlib import Path

LABELS_DIR = Path("data/raw/xbd/train/labels")

def main():
    sample_file = LABELS_DIR / "hurricane-matthew_00000000_post_disaster.json"
    with open(sample_file) as f:
        data = json.load(f)

    print("=== Full metadata dump ===")
    print(json.dumps(data["metadata"], indent=2))

    print("\n=== Sample lng_lat polygon (checking coordinate ranges) ===")
    first_building = data["features"]["lng_lat"][0]
    print(json.dumps(first_building, indent=2))

    # Extract all lng_lat coordinates across every building in this tile to
    # see the real bounding box - this tells us if lng_lat polygons give
    # us genuine world coordinates for this tile's location on Earth.
    all_lngs, all_lats = [], []
    for feat in data["features"]["lng_lat"]:
        wkt = feat["wkt"]
        coords_str = wkt[wkt.index("((") + 2 : wkt.index("))")]
        for pair in coords_str.split(","):
            lng, lat = map(float, pair.split())
            all_lngs.append(lng)
            all_lats.append(lat)

    print(f"\n=== Tile's real-world bounding box (from lng_lat polygons) ===")
    print(f"Longitude range: {min(all_lngs):.6f} to {max(all_lngs):.6f}")
    print(f"Latitude range:  {min(all_lats):.6f} to {max(all_lats):.6f}")
    print("(If these look like plausible real Earth coordinates for the")
    print(" tile's known disaster event location, we have genuine geo data.)")

if __name__ == "__main__":
    main()
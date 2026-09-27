"""
Phase 8: Geospatial damage visualization using real building coordinates
verified in xBD's lng_lat polygon data. Produces an interactive map
showing actual damage locations and severity.
"""
import json
from pathlib import Path

import folium

COLOR_MAP = {
    "no-damage": "green",
    "minor-damage": "yellow",
    "major-damage": "orange",
    "destroyed": "red",
    "un-classified": "gray",
}


def parse_wkt_centroid(wkt: str) -> tuple[float, float]:
    """Returns (lng, lat) centroid of a WKT polygon - simple average of
    vertices, sufficient for building-sized polygons (not geometrically
    exact for complex shapes, but adequate for point-marker placement)."""
    coords_str = wkt[wkt.index("((") + 2 : wkt.index("))")]
    points = [tuple(map(float, pair.split())) for pair in coords_str.split(",")]
    lngs = [p[0] for p in points]
    lats = [p[1] for p in points]
    return sum(lngs) / len(lngs), sum(lats) / len(lats)


def build_damage_map(label_path: Path, output_path: Path):
    with open(label_path) as f:
        data = json.load(f)

    buildings = []
    for feat in data["features"]["lng_lat"]:
        subtype = feat["properties"].get("subtype")
        if subtype is None:
            continue
        lng, lat = parse_wkt_centroid(feat["wkt"])
        buildings.append({"lng": lng, "lat": lat, "subtype": subtype})

    if not buildings:
        print(f"No labeled buildings found in {label_path.name}")
        return

    center_lat = sum(b["lat"] for b in buildings) / len(buildings)
    center_lng = sum(b["lng"] for b in buildings) / len(buildings)

    m = folium.Map(location=[center_lat, center_lng], zoom_start=17, tiles="OpenStreetMap")

    for b in buildings:
        folium.CircleMarker(
            location=[b["lat"], b["lng"]],
            radius=4,
            color=COLOR_MAP.get(b["subtype"], "gray"),
            fill=True,
            fill_color=COLOR_MAP.get(b["subtype"], "gray"),
            fill_opacity=0.8,
            popup=f"Damage: {b['subtype']}",
        ).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Saved map: {output_path} ({len(buildings)} buildings plotted)")

    # Report class breakdown for this specific map
    from collections import Counter
    counts = Counter(b["subtype"] for b in buildings)
    for cls, count in counts.most_common():
        print(f"  {cls}: {count}")
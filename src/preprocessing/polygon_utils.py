"""
Phase 4, Baseline 2: Reusable polygon rasterization and cropping utilities.
Shared by Baseline 2 (feature extraction) and future per-building evaluation.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    coords_str = wkt[wkt.index("((") + 2 : wkt.index("))")]
    return [tuple(map(float, pair.split())) for pair in coords_str.split(",")]


def rasterize_polygon(points: list[tuple[float, float]], image_size: int = 1024) -> np.ndarray:
    """Returns a binary (0/1) mask with the polygon filled, same size as the tile."""
    mask_img = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(points, fill=1)
    return np.array(mask_img, dtype=bool)


def get_buildings_for_tile(label_path: Path) -> list[dict]:
    """
    Returns a list of {uid, subtype, polygon_mask} for every building in a
    post-disaster label file. subtype is None for pre-disaster labels
    (they don't carry damage subtype, per Phase 2 schema verification).
    """
    with open(label_path) as f:
        data = json.load(f)

    buildings = []
    for feat in data["features"]["xy"]:
        props = feat["properties"]
        points = parse_wkt_polygon(feat["wkt"])
        mask = rasterize_polygon(points)
        buildings.append({
            "uid": props["uid"],
            "subtype": props.get("subtype"),
            "polygon_mask": mask,
        })
    return buildings
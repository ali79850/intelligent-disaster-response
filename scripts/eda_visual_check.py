"""
Phase 2, Step 3: Corrupted-image check across all PNGs, plus visual inspection
of one pre/post pair with building polygons overlaid (using the verified
features.xy pixel-coordinate field).
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

IMAGES_DIR = Path("data/raw/xbd/train/images")
LABELS_DIR = Path("data/raw/xbd/train/labels")
OUTPUT_DIR = Path("reports/figures")

def check_corrupted_images():
    print("Checking all images for corruption (this may take a minute)...")
    bad_files = []
    total = 0
    for f in IMAGES_DIR.glob("*.png"):
        total += 1
        try:
            with Image.open(f) as img:
                img.verify()
        except Exception as e:
            bad_files.append((f.name, str(e)))

    print(f"\nChecked {total} images.")
    if bad_files:
        print(f"CORRUPTED/UNREADABLE FILES FOUND: {len(bad_files)}")
        for name, err in bad_files:
            print(f"  {name}: {err}")
    else:
        print("No corrupted files found.")
    return bad_files

def visualize_sample_pair(base_name):
    pre_img_path = IMAGES_DIR / f"{base_name}_pre_disaster.png"
    post_img_path = IMAGES_DIR / f"{base_name}_post_disaster.png"
    post_label_path = LABELS_DIR / f"{base_name}_post_disaster.json"

    with open(post_label_path) as f:
        label = json.load(f)

    color_map = {
        "no-damage": (0, 255, 0),
        "minor-damage": (255, 255, 0),
        "major-damage": (255, 128, 0),
        "destroyed": (255, 0, 0),
        "un-classified": (128, 128, 128),
    }

    post_img = Image.open(post_img_path).convert("RGB")
    draw = ImageDraw.Draw(post_img)

    for feat in label["features"]["xy"]:
        subtype = feat["properties"].get("subtype", "un-classified")
        color = color_map.get(subtype, (255, 255, 255))
        wkt = feat["wkt"]
        coords_str = wkt[wkt.index("((") + 2 : wkt.index("))")]
        points = [tuple(map(float, pair.split())) for pair in coords_str.split(",")]
        draw.polygon(points, outline=color, width=2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{base_name}_post_overlay.png"
    post_img.save(out_path)
    print(f"Saved overlay: {out_path}")

    pre_img = Image.open(pre_img_path)
    pre_out = OUTPUT_DIR / f"{base_name}_pre.png"
    pre_img.save(pre_out)
    print(f"Saved pre-disaster image (no overlay): {pre_out}")

def main():
    check_corrupted_images()
    print("\nGenerating visual sample...")
    visualize_sample_pair("hurricane-matthew_00000000")

if __name__ == "__main__":
    main()
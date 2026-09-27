import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from PIL import Image

from src.inference.engine import DamageInferenceEngine
from src.inference.report import generate_report

def main():
    engine = DamageInferenceEngine(checkpoint_path="models/unet_epoch10.pt")

    base_name = "santa-rosa-wildfire_00000137"
    images_dir = Path("data/raw/xbd/train/images")
    pre_img = Image.open(images_dir / f"{base_name}_pre_disaster.png")
    post_img = Image.open(images_dir / f"{base_name}_post_disaster.png")

    result = engine.predict(pre_img, post_img)
    report = generate_report(result, tile_id=base_name)

    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
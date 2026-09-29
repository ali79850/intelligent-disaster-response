"""
Phase 11: Verify LLM summary generation against a known report, checking
specifically that it does NOT invent forbidden facts (casualties,
population, infrastructure status).
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from PIL import Image

from src.inference.engine import DamageInferenceEngine
from src.inference.report import generate_report
from src.llm.summarizer import generate_narrative_summary

def main():
    engine = DamageInferenceEngine(checkpoint_path="models/unet_epoch10.pt")

    base_name = "hurricane-matthew_00000000"
    images_dir = Path("data/raw/xbd/train/images")
    pre_img = Image.open(images_dir / f"{base_name}_pre_disaster.png")
    post_img = Image.open(images_dir / f"{base_name}_post_disaster.png")

    result = engine.predict(pre_img, post_img)
    report = generate_report(result, tile_id=base_name)

    print("=== Structured report (input to LLM) ===")
    print(f"Affected area: {report['summary']['affected_area_percentage']}%")
    print(f"Requires review: {report['quality_flags']['requires_human_review']}")

    print("\n=== LLM narrative summary ===")
    summary = generate_narrative_summary(report)
    print(summary)

    print("\n=== Manual check ===")
    forbidden_terms = ["casualt", "death", "injur", "popul", "hospital", "rescue", "evacuat"]
    found = [t for t in forbidden_terms if t in summary.lower()]
    if found:
        print(f"WARNING: possible fabricated content, contains: {found}")
    else:
        print("No forbidden terms found - looks clean.")

if __name__ == "__main__":
    main()
"""
Phase 12: Regression test for the inference engine against a known,
extensively-verified result (hurricane-matthew_00000000, destroyed class
= 61130 pixels - verified independently via direct script call, curl API
request, and browser UI across Phases 9-11). If this test ever fails, it
means something changed in preprocessing, normalization, or the model
checkpoint itself - a real, meaningful signal, not a flaky test.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
import pytest
from PIL import Image

from src.inference.engine import DamageInferenceEngine

MODEL_PATH = "models/unet_epoch10.pt"


@pytest.fixture(scope="module")
def engine():
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model checkpoint not found at {MODEL_PATH} - skipping model-dependent tests")
    return DamageInferenceEngine(checkpoint_path=MODEL_PATH)


def test_known_tile_destroyed_pixel_count(engine):
    images_dir = Path("data/raw/xbd/train/images")
    base_name = "hurricane-matthew_00000000"
    pre_img = Image.open(images_dir / f"{base_name}_pre_disaster.png")
    post_img = Image.open(images_dir / f"{base_name}_post_disaster.png")

    result = engine.predict(pre_img, post_img)

    assert result["class_statistics"]["destroyed"]["pixel_count"] == 61130


def test_output_shape_and_structure(engine):
    images_dir = Path("data/raw/xbd/train/images")
    base_name = "hurricane-matthew_00000000"
    pre_img = Image.open(images_dir / f"{base_name}_pre_disaster.png")
    post_img = Image.open(images_dir / f"{base_name}_post_disaster.png")

    result = engine.predict(pre_img, post_img)

    assert result["predicted_mask"].shape == (1024, 1024)
    assert result["class_probabilities"].shape == (5, 1024, 1024)
    assert result["model_version"] == "unet_epoch10"
    assert 0.0 <= result["mean_confidence"] <= 1.0


def test_class_statistics_sum_to_total_pixels(engine):
    images_dir = Path("data/raw/xbd/train/images")
    base_name = "hurricane-matthew_00000000"
    pre_img = Image.open(images_dir / f"{base_name}_pre_disaster.png")
    post_img = Image.open(images_dir / f"{base_name}_post_disaster.png")

    result = engine.predict(pre_img, post_img)

    total = sum(stat["pixel_count"] for stat in result["class_statistics"].values())
    assert total == 1024 * 1024
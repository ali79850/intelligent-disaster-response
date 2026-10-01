"""
Phase 12: API-level tests using FastAPI's TestClient. Tests request
validation and error handling; the full /api/analyze happy path is
covered by Phase 10's manual end-to-end verification (documented in
DECISIONS.md) and is exercised here too, skipping gracefully if the
model checkpoint is unavailable.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "backend"))
from main import app

MODEL_PATH = "models/unet_epoch10.pt"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "unet_epoch10"
    assert len(data["class_names"]) == 5
    assert "known_limitations" in data
    assert len(data["known_limitations"]) > 0


def test_analyze_rejects_non_image_file(client):
    response = client.post(
        "/api/analyze",
        files={
            "pre_image": ("test.txt", b"not an image", "text/plain"),
            "post_image": ("test.txt", b"not an image", "text/plain"),
        },
    )
    assert response.status_code == 400


def test_analyze_rejects_mismatched_image_sizes(client):
    import io
    from PIL import Image

    small_img = io.BytesIO()
    Image.new("RGB", (100, 100)).save(small_img, format="PNG")
    small_img.seek(0)

    large_img = io.BytesIO()
    Image.new("RGB", (200, 200)).save(large_img, format="PNG")
    large_img.seek(0)

    response = client.post(
        "/api/analyze",
        files={
            "pre_image": ("pre.png", small_img, "image/png"),
            "post_image": ("post.png", large_img, "image/png"),
        },
    )
    assert response.status_code == 400
    assert "size mismatch" in response.json()["detail"].lower()


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not available")
def test_analyze_known_tile_end_to_end(client):
    images_dir = Path("data/raw/xbd/train/images")
    base_name = "hurricane-matthew_00000000"

    with open(images_dir / f"{base_name}_pre_disaster.png", "rb") as pre_f, \
         open(images_dir / f"{base_name}_post_disaster.png", "rb") as post_f:
        response = client.post(
            "/api/analyze",
            files={
                "pre_image": ("pre.png", pre_f, "image/png"),
                "post_image": ("post.png", post_f, "image/png"),
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["damage_distribution"]["destroyed"]["pixel_count"] == 61130
    assert data["quality_flags"]["requires_human_review"] is True
"""
Phase 10: FastAPI backend wrapping the verified inference engine and report
generator (Phase 9). Model loads once at startup, never per-request, per
project rule P (version the model, separate training from inference).
"""
import io
import sys
from pathlib import Path
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference.engine import DamageInferenceEngine
from src.inference.report import generate_report

app = FastAPI(
    title="Intelligent Disaster Response & Damage Assessment API",
    description=(
        "AI-assisted preliminary damage assessment from pre/post-disaster "
        "imagery. This is a decision-support tool, not an authoritative "
        "emergency response system. See /api/model-info for model details "
        "and documented limitations."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded ONCE at server startup - not per-request
MODEL_CHECKPOINT = str(PROJECT_ROOT / "models" / "unet_epoch10.pt")
CONFIG_PATH = str(PROJECT_ROOT / "configs" / "config.yaml")
engine: DamageInferenceEngine | None = None


@app.on_event("startup")
def load_model():
    global engine
    engine = DamageInferenceEngine(checkpoint_path=MODEL_CHECKPOINT, config_path=CONFIG_PATH)
    print(f"Model loaded: {engine.model_version} on {engine.device}")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.get("/api/model-info")
def model_info():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": engine.model_version,
        "device": str(engine.device),
        "num_classes": 5,
        "class_names": ["background", "no-damage", "minor-damage", "major-damage", "destroyed"],
        "known_limitations": [
            "Severity discrimination is weaker than damage localization - "
            "see project error analysis. Model tends toward lower confidence "
            "on no-damage, minor-damage, and major-damage classes specifically.",
            "Trained and evaluated on the xBD dataset; performance on imagery "
            "from different sensors, resolutions, or regions not represented "
            "in training is not verified.",
        ],
    }


@app.post("/api/analyze")
async def analyze(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    for f in (pre_image, post_image):
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a valid image")

    try:
        pre_bytes = await pre_image.read()
        post_bytes = await post_image.read()
        pre_pil = Image.open(io.BytesIO(pre_bytes))
        post_pil = Image.open(io.BytesIO(post_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image data: {e}")

    if pre_pil.size != post_pil.size:
        raise HTTPException(
            status_code=400,
            detail=f"Pre/post image size mismatch: {pre_pil.size} vs {post_pil.size}",
        )

    result = engine.predict(pre_pil, post_pil)
    report = generate_report(result, tile_id=pre_image.filename or "uploaded_tile")
    return report

from src.llm.summarizer import generate_narrative_summary


class SummarizeRequest(BaseModel):
    report: dict


@app.post("/api/summarize")
async def summarize(request: SummarizeRequest):
    try:
        summary = generate_narrative_summary(request.report)
        return {"summary": summary}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM summary generation failed: {e}")

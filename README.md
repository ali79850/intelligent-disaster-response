# Intelligent Disaster Response & Damage Assessment Platform

An AI-assisted platform for analyzing pre- and post-disaster satellite imagery to produce structured, building-level damage assessments — with a working inference API, interactive dashboard, geospatial visualization, and a disciplined LLM summary layer.

**This is a research and portfolio project.** It is a decision-support and analytical tool, not a replacement for professional emergency responders, structural engineers, or official disaster assessment authorities. Every prediction is preliminary and may be wrong — see [Limitations](#limitations).

---

## Overview

Given a pre-disaster and post-disaster satellite image pair, this system:
- Detects and localizes changed/damaged buildings
- Classifies damage severity per pixel (no-damage, minor, major, destroyed)
- Reports confidence per class and flags low-confidence results for human review
- Visualizes results on an interactive map using the imagery's real geographic coordinates
- Optionally generates a plain-language summary of the findings via an LLM, strictly constrained to the computed numbers

## Problem

Rapid, consistent damage assessment after a disaster is a real bottleneck in emergency response. Manual assessment from satellite imagery is slow and resource-intensive. This project explores how far a modest, honestly-evaluated deep learning pipeline can go toward automating a first-pass assessment — while being explicit about where it fails.

## Dataset

**[xBD (xView2)](https://xview2.org)** — the standard academic benchmark for building damage assessment, chosen over alternatives (RescueNet, FloodNet, AIDER) specifically because it is the only dataset providing genuine paired pre/post-disaster imagery with building-level polygon annotations on an ordinal damage scale. Full selection rationale in [`DECISIONS.md`](DECISIONS.md).

- **Scale used:** Challenge training set — 2,799 tiles (1024×1024, ~7.8 GB), 2,799 pre/post pairs, 162,787 labeled building instances
- **Damage classes (verified from actual data, not assumed):** `no-damage`, `minor-damage`, `major-damage`, `destroyed`, `un-classified` — 5 classes, not the 4 initially assumed from the paper description alone
- **License:** CC BY-NC-SA 3.0 (non-commercial). Not included in this repository — see [Reproducibility](#reproducibility) for acquisition.
- **Real-world georeferencing confirmed:** building polygons carry genuine lat/lon coordinates (verified against Hurricane Matthew's actual October 2016 Haiti landfall location and date)

### Class Imbalance (measured, not assumed)
Pixel-level distribution across all tiles: **94.12% background**, 4.30% no-damage, 0.53% minor-damage, 0.68% major-damage, 0.32% destroyed. This imbalance is severe and directly shaped every loss-function and evaluation-metric decision in this project — plain pixel accuracy is never used as a headline metric.

## Methodology

### Train/Val/Test Split
**Event-based**, not random-by-tile — deliberately designed to test generalization to *unseen disaster types*, not just unseen images:
- **Train** (63.7%): socal-fire, hurricane-michael, hurricane-florence, midwest-flooding, guatemala-volcano
- **Val** (15.7%): hurricane-harvey, mexico-earthquake (earthquake type unseen in train)
- **Test** (20.6%): hurricane-matthew, santa-rosa-wildfire, palu-tsunami (tsunami type unseen in train)

This was a deliberate trade-off: it produces a harder, more honest evaluation than a random split, at the cost of imperfect class balance across splits (documented in `DECISIONS.md`).

### Preprocessing
- Dataset-specific normalization statistics computed from the training split (not ImageNet defaults — satellite imagery has a measurably different intensity distribution)
- Segmentation masks rasterized directly from building polygons, verified pixel-for-pixel against visual overlays
- Geometric augmentation only (flips, 90° rotations), deliberately excluding scale-altering augmentation given measured ground-sample-distance variance (1.24–3.15 m/pixel) across tiles
- `un-classified` buildings excluded from loss/metrics via an ignore-index, not silently erased from visualizations

## Model Architecture

**U-Net with a ResNet-18 encoder (ImageNet-pretrained) and skip connections**, taking concatenated pre+post images (6 input channels) and producing a 5-class per-pixel prediction.

This architecture was not chosen by default — it was the direct, evidence-based response to a documented baseline failure (see [Baselines](#baselines-and-model-comparison) below).

**Loss:** combined weighted cross-entropy + Dice loss, addressing the severe class imbalance from two complementary angles (per-pixel weighting and region-overlap optimization).

**Training:** 10 epochs total (5 + 5 resumed), batch size 4, Adam (lr=1e-4), on a Google Colab T4 GPU — local CPU training was measured and confirmed infeasible (~4 hours/epoch) before moving to GPU.

## Baselines and Model Comparison

Three baselines were built before the final model, each evaluated honestly — including when results were poor:

| Model | Test Macro F1 | Key Finding |
|---|---|---|
| Baseline 1: Naive pixel-difference | N/A (binary only, precision 0.039) | 25:1 false-positive ratio — confirms raw pixel differencing cannot separate real damage from shadows, sun-angle variation, and registration noise |
| Baseline 2: Random forest, hand-crafted features | 0.31 | No-damage learnable; all real damage classes weak; feature importances nearly uniform, suggesting the feature *type* (not just choice) is limiting |
| Baseline 3: Simple CNN, no skip connections | 0.27 | **Complete collapse** — never predicts any damage severity class, only background/no-damage |
| **U-Net + ResNet-18 + skip connections (final)** | **0.3380** | Beats all baselines; the only model that successfully localizes damage without collapsing |

Baseline 3's complete failure — traced to aggressive downsampling with no skip connections destroying fine spatial detail — is the direct, documented justification for the final architecture's skip connections.

## Results and Error Analysis

The final model's weakness was characterized through three independent, converging lines of evidence:

1. **Confusion matrix (quantitative):** destroyed class has high recall (0.76) but low precision (0.17) — the model over-predicts severe damage broadly
2. **Visual error inspection (qualitative):** on a specific misclassified tile, the model's predicted "destroyed" region sits in the *exact same location* as ground-truth "minor-damage" buildings — confirming correct spatial localization but confused severity
3. **Grad-CAM explainability:** targeted gradient attribution for the same wrong prediction confirms the model's internal attention genuinely focuses on the real buildings, not an irrelevant visual artifact

**A second, distinct failure mode was discovered through live dashboard use** (not a pre-planned test): on a fire-disaster tile with only 2 ground-truth buildings dominated by visible burned vegetation, the model flagged ~5,000 "destroyed" pixels but only 53 genuinely overlapped the real buildings — it appears to react to general burn/vegetation-damage visual signals, not specifically building destruction. This is documented in full in `DECISIONS.md` as a genuine, organically-found limitation.

**Refined diagnosis (from the deployed report-generation layer):** per-class confidence breakdown shows the model is specifically, measurably uncertain about the three *middle* severity classes (no-damage, minor, major — confidence 0.25–0.48) while confidently handling the extremes (background 0.97, destroyed 0.82, though sometimes wrongly). This is a more precise characterization than "severity miscalibration" alone.

## Explainability

Grad-CAM, adapted for dense per-pixel segmentation output (standard Grad-CAM assumes a single classification score; here, gradients are backpropagated from a specific class's logits summed over a specific region of interest, enabling explanation of one particular prediction rather than a generic class-level heatmap).

**What this shows:** which spatial regions had the greatest gradient influence on a target prediction at the chosen layer. **What this does not prove:** causal necessity, or the contribution of the decoder/skip-connection pathway not captured by the single hooked layer.

## Geospatial Analysis

Building-level damage markers are plotted on a real interactive map (Folium/Leaflet) using the dataset's genuine geographic coordinates — verified against the actual landfall location and date of Hurricane Matthew (Haiti, October 2016). Not a simplified or synthetic visualization.

## System Architecture
Training (Colab GPU)
↓
Saved model checkpoint (models/unet_epoch10.pt)
↓
Inference engine (src/inference/engine.py) — loads once, never retrains
↓
Report generator (src/inference/report.py) — structured JSON, real numbers only
   ↓
┌──┴──┐
↓     ↓
FastAPI Optional LLM summary (Groq API, strictly numbers-in/prose-out,
backend verified to never fabricate casualties/population/infrastructure)
↓
React + TypeScript + Tailwind dashboard

Training and inference are fully separated, per design: the API never retrains on request, and loads the model checkpoint exactly once at startup.

## Dashboard

React + TypeScript + Vite + Tailwind CSS. Features: drag-and-drop image upload with preview, damage distribution visualization, per-class confidence display with low-confidence highlighting, a prominent human-review flag, an optional AI-generated plain-language summary, and a collapsible (never hidden) limitations section. Designed with explicit HCI principles: visibility of system status, error prevention, accessible redundant coding (never color alone), and progressive disclosure.

## API

FastAPI backend, model loaded once at startup via a lifespan context manager.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/model-info` | GET | Model version, classes, known limitations |
| `/api/analyze` | POST | Upload pre/post images, get a full structured damage report |
| `/api/summarize` | POST | Generate a plain-language narrative from an existing report (optional, separate from core analysis) |

## Testing

19 automated tests (`pytest`) across four layers::
- Pure logic (segmentation metrics math, report-generation aggregation) — hand-calculated expected values, not placeholder assertions
- The real trained model — a regression test locks in an exact, independently-verified prediction value
- The full API layer via FastAPI's `TestClient`, including validation-rejection cases (not just happy paths)

Run with:
```powershell
pytest tests\ -v
```

## Installation & Usage

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for complete, tested, step-by-step setup — environment, dataset acquisition, preprocessing, training, and running the full application. Summary:

```powershell
# Environment
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

# Backend (after dataset/model setup per REPRODUCIBILITY.md)
cd app\backend
uvicorn main:app --reload

# Frontend (separate terminal)
cd app\frontend
npm install
npm run dev
```

## Limitations

Stated plainly, not minimized:

- **Severity discrimination is weaker than damage detection.** The model reliably finds where buildings are and whether something changed, but is measurably less confident distinguishing degrees of damage among the three middle severity classes.
- **A documented vegetation/burn-confusion failure mode exists** for fire-disaster tiles with sparse building coverage and dominant landscape change.
- **Trained and evaluated only on xBD.** Performance on imagery from different sensors, resolutions, or geographic regions not represented in training is not verified.
- **`un-classified` ground-truth regions were excluded from training**, not resolved — the model has no signal for genuinely ambiguous cases.
- **This tool does not and cannot determine casualties, population impact, or infrastructure operational status.** The LLM summary layer is explicitly, verifiably constrained to never state these.
- **Non-commercial dataset license (CC BY-NC-SA 3.0)** restricts this project to research/portfolio use.

## Future Work

- Targeted experiment adjusting class-weighting/loss-balance to directly test whether the destroyed-class over-prediction bias can be reduced (documented hypothesis, not yet run)
- Incorporating xBD's tier3 supplemental data for greater disaster-event diversity (same verified schema, no cross-dataset reconciliation risk)
- Addressing the vegetation/burn-confusion failure mode, potentially via explicit vegetation-masking preprocessing
- A second dataset (e.g., RescueNet) for genuine cross-dataset generalization testing — a larger undertaking, deliberately deferred

## Project Structure
intelligent-disaster-response/
├── app/
│ ├── backend/ # FastAPI application
│ └── frontend/ # React + TypeScript + Vite + Tailwind
├── configs/ # config.yaml — paths, normalization stats
├── data/ # raw/interim/processed (not committed; see REPRODUCIBILITY.md)
├── models/ # trained checkpoints (not committed; too large for git)
├── reports/figures/ # error analysis visualizations, Grad-CAM outputs
├── scripts/ # one-off analysis, verification, and debugging scripts
├── src/
│ ├── data/ # PyTorch Dataset
│ ├── evaluation/ # segmentation metrics
│ ├── explainability/# Grad-CAM
│ ├── geospatial/ # damage mapping
│ ├── inference/ # engine + report generation
│ ├── llm/ # narrative summary layer
│ ├── models/ # baseline CNN, U-Net
│ ├── preprocessing/ # mask generation, polygon utilities
│ └── training/ # loss functions
├── tests/ # pytest suite
├── DECISIONS.md # full technical decision log with reasoning
└── REPRODUCIBILITY.md # tested, step-by-step setup instructions

## License

Code: MIT (see `LICENSE`). Dataset: CC BY-NC-SA 3.0, obtained separately per [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — not redistributed in this repository.
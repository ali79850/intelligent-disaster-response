# Technical Decision Log

## 2026-09 — Phase 0: Dataset Selection

**Decision:** Use xBD (xView2) as the primary dataset.
**Alternatives considered:** RescueNet, FloodNet, AIDER.
**Reason:** Only candidate dataset with genuine paired pre-/post-disaster imagery
AND building-level polygon annotations on an ordinal 4-class damage scale.
Required for change detection, segmentation, and severity classification to all
be legitimate on a single dataset.
**Trade-offs:** Non-commercial license (CC BY-NC-SA 3.0) restricts the project to
research/portfolio use. Lacks native scene-level classes (water, debris, blocked
roads) — deferred as a possible Phase 2+ extension via RescueNet.
**Status:** Confirmed pending dataset access verification.

## 2026-09 — Phase 1: Repository Foundation

**Decision:** Use a clean, space-free repo path and a minimal Phase-1 requirements.txt,
adding heavier dependencies (PyTorch, OpenCV, GeoPandas) only in the phases that need them.
**Reason:** Avoid path-escaping bugs later; avoid unused/unjustified dependencies.
**Status:** Confirmed.

**Note:** Development environment uses Python 3.13 (cp313 wheels observed during
pip install). PyTorch/GeoPandas/Rasterio compatibility with 3.13 will be verified
explicitly, not assumed, when those dependencies are introduced (Phases 5 and 8).

## 2026-09 — Phase 1: Dataset Download Method

**Decision:** Download xBD Challenge training set via `curl` in a single
continuous request rather than the browser's download manager.
**Reason:** The signed download URL (AWS CloudFront, time-limited `Expires`
parameter) failed with HTTP 403 "Forbidden" when the browser paused/resumed
the transfer near completion (~96%). A single uninterrupted `curl` request
avoided any resume attempt and completed cleanly, hash-verified against the
published SHA1.
**Status:** Confirmed. Verified structure: train/images, train/labels,
train/targets, 5,598 files each (2,799 pre/post pairs).
## 2026-09 — Phase 2: Label Schema Verified

**Finding:** Actual xBD post-disaster labels contain 5 distinct `subtype` values,
not 4: `no-damage`, `minor-damage`, `major-damage`, `destroyed`, `un-classified`.
Verified by scanning all 2,799 post_disaster label JSONs in the downloaded
training set.
**Impact:** Phase 0 scope described 4 damage categories based on the general
xBD paper description. The `un-classified` category's prevalence and handling
(treat as 5th class vs. exclude vs. separate uncertainty analysis) will be
decided based on its actual frequency, determined in the next EDA step.
**Status:** Open — pending class distribution counts.

## 2026-09 — Phase 2: Class Distribution & Imbalance (verified)

**Finding:** Across 162,787 building instances in 2,799 post-disaster tiles:
no-damage 72.13%, minor-damage 9.20%, major-damage 8.70%, destroyed 8.13%,
un-classified 1.84%. Distribution varies drastically by disaster event
(e.g. mexico-earthquake: 99.36% no-damage vs. hurricane-matthew: 18.04%
no-damage) — disaster-event identity is a strong confound.
**Impact:**
1. Severe class imbalance requires weighted/focal loss in Phase 5, and
   precision/recall/F1/IoU over accuracy in Phase 6 evaluation.
2. Train/val/test split (Phase 3) must consider holding out entire disaster
   events, not random tiles, to prevent the model learning event-identity
   shortcuts instead of visual damage features. Final split strategy to be
   decided in Phase 3 with full reasoning documented.
3. un-classified (1.84% overall) will likely be excluded from training/
   evaluation as a simplification, pending a closer look — this is a
   decision, not a default, and will be recorded separately when made.
**Status:** Confirmed via full scan of all 2,799 post-disaster label files.
## 2026-09 — Phase 2: Coordinate/Polygon Alignment Verified

**Finding:** Visual inspection of hurricane-matthew_00000000_post_overlay.png
confirms features.xy WKT polygons correctly align with real building footprints
in the image (polygons trace the actual settlement cluster, not scattered over
empty fields). Also confirmed: building density varies enormously by tile —
this sample tile is a sparse rural riverside settlement, most of the frame is
empty agricultural land.
**Impact:** Confirms our WKT-to-pixel-polygon parsing logic is correct before
building mask-generation code in Phase 3. Building density variation across
tiles is a factor to consider in patch sampling strategy.
**Status:** Confirmed via manual visual inspection.
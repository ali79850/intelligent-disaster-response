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
## 2026-09 — Phase 2: Image Dimensions & GSD Verified

**Finding:** All 5,598 images (pre + post combined) are uniformly 1024x1024
pixels — confirmed via full scan of all label metadata, not assumed. Ground
sample distance (gsd) varies from 1.24 to 3.15 m/pixel (avg 2.14) across tiles,
meaning real-world object scale is not constant across the dataset despite
uniform pixel dimensions. Disaster events also group into 6 broader
disaster_type categories: fire, flooding, wind, earthquake, tsunami, volcano.
**Impact:** No resizing/padding needed for uniform input size in Phase 3.
GSD variance is a documented limitation — noted for potential normalization
or as a discussed caveat in error analysis (Phase 6), not addressed now.
**Status:** Confirmed via full scan of all 5,598 label files.
## 2026-09 — Phase 3: Train/Val/Test Split Strategy

**Decision:** Event-based (grouped) split rather than random tile split.
- Train: socal-fire, hurricane-michael, hurricane-florence, midwest-flooding,
  guatemala-volcano (1,782 tiles, 63.7%)
- Val: hurricane-harvey, mexico-earthquake (440 tiles, 15.7%)
- Test: hurricane-matthew, santa-rosa-wildfire, palu-tsunami (577 tiles, 20.6%)
**Reason:** Phase 2 EDA showed disaster-event identity strongly correlates with
damage-class distribution (e.g. mexico-earthquake 99% no-damage vs.
hurricane-matthew 18%). A random tile split risks the model learning
event-identity shortcuts instead of visual damage cues. This split ensures
val (earthquake) and test (tsunami) each include a disaster_type never seen
in train, directly testing generalization to unseen disaster types.
**Trade-offs:** Train is 63.7% of tiles, below the ~70% target, due to event
granularity constraints. guatemala-volcano (18 tiles) is too small to serve
as a reliable held-out set, so volcano-type damage is never evaluated in
val/test — a documented limitation, not an oversight. Class balance across
splits will differ since grouping is event-based, not stratified by damage
class — to be measured, not assumed, in the next step.
**Status:** Approved by user.
## 2026-09 — Phase 3: Split Class Imbalance — Kept As-Is

**Finding:** Generated splits show val set has only 0.73% destroyed-class
buildings (403 instances) vs. 18.16% in test — a direct consequence of val
including mexico-earthquake (99.36% no-damage per Phase 2 EDA).
**Decision:** Keep the event-based split unchanged rather than reshuffle
events to balance class distribution.
**Reason:** Reshuffling to produce a more convenient class balance would
undermine the split's purpose (honest generalization testing) and risks
being a form of metric-shopping. The imbalance is a real, disclosable
property of testing against an authentic unseen disaster event.
**Mitigation:** Val-set destroyed-class metrics will be treated as low-
confidence signals during training (too few examples for stability);
macro/weighted F1 and full test-set results will be the primary basis for
model comparison, not per-class val curves in isolation. This caveat will
be documented in the final results/limitations section.
**Status:** Confirmed.
## 2026-09 — Phase 3: Un-classified Building Handling

**Decision:** Rasterize un-classified buildings (1.84% of all instances) as
their own pixel value in segmentation masks, but exclude them from loss
computation and evaluation metrics via a pixel-level ignore mask.
**Alternatives considered:**
- Treat as a 6th real class the model must learn — rejected because
  un-classified reflects human annotator uncertainty, not a distinct visual
  damage pattern, and would add noisy/unlearnable signal to training.
- Rasterize as background — rejected because it silently erases real
  buildings from the mask, which could mislead debugging/visualization later
  (a building appears to not exist rather than "exists, unknown label").
**Reason for chosen option:** Keeps ground truth visually honest (all real
buildings appear in the mask) while not penalizing or rewarding the model
for a category that isn't a principled learning target.
**Class-to-pixel-value mapping (final):**
  0 = background, 1 = no-damage, 2 = minor-damage, 3 = major-damage,
  4 = destroyed, 5 = un-classified (ignored in loss/metrics via ignore mask)
**Status:** Confirmed.
## 2026-09 — Phase 3: Pixel-Level Class Distribution (Segmentation Masks)

**Finding:** Generated segmentation masks for all 2,799 post-disaster tiles.
Pixel-level distribution: background 94.12%, no-damage 4.30%, minor-damage
0.53%, major-damage 0.68%, destroyed 0.32%, un-classified 0.05%. This is far
more extreme than the building-instance-level distribution from Phase 2
(no-damage 72.13% there) — most of a satellite tile's area is non-building
ground (roads, fields, water), not damage-relevant pixels.
**Impact:** Confirms plain per-pixel cross-entropy loss is unusable — a
model predicting all-background achieves ~94% pixel accuracy while learning
nothing. Phase 5 will require class-weighted loss and/or Dice/focal loss,
and Phase 6 evaluation must report per-class IoU/F1, never overall pixel
accuracy as a headline metric.
**Status:** Confirmed via full mask generation over all 2,799 tiles.
## 2026-09 — Phase 3: Mask Generation Visually Verified

**Finding:** Blended overlay of generated mask on source image
(hurricane-matthew_00000000) confirms mask regions align exactly with the
same building cluster verified in the earlier polygon overlay check.
Colors match expected damage profile (predominantly minor-damage/yellow,
some destroyed/red). Individual building shapes are distinct, not merged -
confirms polygon rasterization has no overlap/fill-rule bug.
**Status:** Confirmed via visual inspection. Mask generation pipeline
verified end-to-end (schema -> coordinates -> rasterization -> visual check).
## 2026-09 — Phase 3: Pair Completeness Verified

**Finding:** Validated all 2,799 tiles across train (1,782), val (440), and
test (577) splits have complete pre-image, post-image, post-disaster label,
and generated mask files. Zero missing files found.
**Status:** Confirmed via full scan. Safe to proceed to Dataset/DataLoader
implementation.
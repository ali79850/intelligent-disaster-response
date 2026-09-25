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
## 2026-09 — Phase 3: Dataset Normalization Statistics Computed

**Decision:** Use dataset-specific mean/std (computed from train split, pre+
post images combined) for normalization, rather than ImageNet statistics.
**Computed values (RGB, [0,1] scale):**
  mean: [0.2920, 0.3253, 0.2438]
  std:  [0.1571, 0.1397, 0.1314]
**Reason:** Satellite imagery has a measurably different intensity/color
distribution than ImageNet's natural photographs (our means are notably
lower/darker). Computed only from train split to avoid val/test leakage
into normalization statistics.
**Impact:** These constants will be stored in configs/config.yaml and
referenced from there in the Dataset class — not hardcoded inline.
**Status:** Confirmed.## 2026-09 — Phase 3: PyTorch Installed (CPU-only)

**Decision:** Install PyTorch CPU build via
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
**Verification:** Some older community reports suggested PyTorch lacked
Python 3.13 wheel support on Windows; this was checked directly rather than
assumed. Install succeeded cleanly: torch 2.14.0+cpu, imports correctly,
CUDA unavailable as expected (no dedicated GPU on this machine).
**Impact:** Local development/training will run CPU-only, consistent with
Phase 0's hardware plan. GPU training (Colab/Kaggle) remains available for
Phase 5 if local CPU training proves too slow for full-scale experiments.
**Status:** Confirmed.
## 2026-09 — Phase 3: XBDDataset Class Implemented and Verified

**Finding:** src/data/xbd_dataset.py implemented and verified against all
three splits: correct sample counts (1782/440/577), correct tensor shapes
((3,1024,1024) images, (1024,1024) masks), correct dtypes (float32/int64),
and confirmed un-classified remapping (raw value 5 -> ignore_index 255)
via guatemala-volcano_00000025, a tile known to contain un-classified
buildings from Phase 2 EDA.
**Status:** Confirmed. Dataset class ready for DataLoader integration.
## 2026-09 — Phase 3: Augmentation Strategy

**Decision:** Geometric augmentation only (horizontal flip, vertical flip,
90-degree rotations), applied identically to pre-image/post-image/mask via
a single shared random choice per sample. Train split only.
**Reason:** These transforms preserve real-world scale exactly, avoiding
interaction with the GSD variance (1.24-3.15 m/pixel) documented earlier.
Scale-altering augmentations (random resized crop, zoom) deliberately
excluded for now to avoid distorting apparent building size.
**Deferred:** Color/photometric augmentation (brightness, contrast) - the
dataset has real sun angle/sensor metadata variation worth analyzing first,
rather than picking augmentation ranges without evidence. Revisit in
Phase 5 experiments if needed.
**Status:** Confirmed.
## 2026-09 — Phase 3: Augmentation Verified

**Finding:** Confirmed augmentation randomizes correctly (6 distinct mask
variants across 10 calls to the same dataset index) and preserves tensor
shapes. Confirmed val split with augment=False loads without augmentation
applied, as intended.
**Status:** Confirmed. Dataset class with augmentation support is complete
and verified end-to-end.
## 2026-09 — Phase 3: DataLoader Implemented and Verified

**Finding:** DataLoader with batch_size=2, num_workers=0, shuffle=True (train)
verified: correct batch dimension on all tensors ((2,3,1024,1024) images,
(2,1024,1024) masks), correct shuffling (distinct random base names per
batch). Measured throughput: ~0.50s/batch data loading alone, implying
~7.4 minutes/epoch for data loading on this CPU-only setup before any
model computation is added.
**Impact:** This is a real, measured baseline for planning Phase 5 training
time budgets - not an assumption. If model forward/backward pass adds
significantly more time per batch, full training runs may need to move to
Colab/Kaggle GPU rather than local CPU, per the Phase 0 hardware plan.
**Status:** Confirmed. Phase 3 (preprocessing & data pipeline) complete.
## 2026-09 — Phase 4: Polygon Utility Scope

**Decision:** Build polygon-to-pixel-mask/crop utilities as reusable code
under src/preprocessing/, not scoped narrowly to Baseline 2 only.
**Reason:** Per-building feature extraction is needed for Baseline 2 now,
and will be needed again for per-building evaluation/statistics in later
phases (building-level damage reporting, per your original project scope).
Building it once, reusably, avoids duplicating polygon-parsing logic.
**Status:** Confirmed.
## 2026-09 — Phase 4: Baseline 1 (Naive Pixel-Diff) — Results

**Method:** Absolute grayscale pixel difference between pre/post images,
threshold=30, evaluated as binary change detection against collapsed
ground truth (no-damage vs. any real damage).
**Results (test set, 577 tiles):** Precision 0.0386, Recall 0.4972,
F1 0.0716, IoU 0.0372. TP=6,023,700 FP=150,021,694 FN=6,091,284
TN=442,481,569.
**Interpretation:** Extremely low precision (25:1 false-positive ratio)
confirms raw pixel differencing cannot distinguish real damage from
confounds like shadows, sun-angle variation between capture dates,
vegetation change, and imperfect image registration. Recall near 50%
shows damage does cause some detectable pixel change, but the signal is
overwhelmed by noise. This result is reported honestly as the floor to
beat — poor baseline performance here is expected and useful, not hidden.
**Status:** Confirmed. Establishes Baseline 1 floor for later comparison.
## 2026-09 — Phase 4: Polygon Utility Verified

**Finding:** src/preprocessing/polygon_utils.py verified against
hurricane-matthew_00000000 - correctly extracts 105 buildings with
non-zero pixel counts and correct subtype labels, consistent with prior
visual verification of this same tile (predominantly minor-damage).
**Status:** Confirmed. Ready for feature extraction (Baseline 2).
## 2026-09 — Phase 4: Feature Extraction — Performance Bug Found and Fixed

**Finding:** Initial extract_building_features.py implementation computed
Sobel edge detection over the full 1024x1024 image separately for every
building in a tile, rather than once per tile. On dense tiles (500+
buildings), this caused the script to hang for 8+ hours with no progress
on val/test splits after train completed normally.
**Fix:** Refactored to compute the Sobel edge map once per image (pre and
post), then index into it per building via the polygon mask - same pattern
already correctly used for mean_diff/mean_pre/mean_post.
**Result:** val (440 tiles, 54,921 buildings) and test (577 tiles, 57,045
buildings) completed in minutes after the fix. Total: 159,794 buildings
with features extracted (train 47,828 + val 54,921 + test 57,045),
consistent with ~163k total buildings minus ~3k excluded un-classified.
**Status:** Confirmed. Lesson: watch for per-item recomputation of
image-level operations inside per-building/per-polygon loops.
**Impact for the record - the process was left running overnight before
being diagnosed; no data was lost since train had already completed and
was written incrementally, but this cost real wall-clock time. Worth
building smaller smoke-test runs (e.g., 10 tiles) before launching a full
dataset pass on new preprocessing code going forward.**
## 2026-09 — Phase 4: Baseline 2 (Random Forest) — Results

**Method:** Random forest (200 trees, class_weight='balanced') on 8
hand-crafted per-building features (grayscale diff stats, area, edge
density), predicting 4-class damage severity.
**Results (test set, 57,045 buildings):** Accuracy 0.61, macro F1 0.31,
weighted F1 0.58. Per-class F1: no-damage 0.78, destroyed 0.26,
minor-damage 0.15, major-damage 0.06.
**Interpretation:** No-damage is learnable from simple statistics; all real
damage classes are weak. Feature importances are nearly uniform (0.117-
0.137 across all 8 features) - no single hand-crafted feature dominates,
suggesting these simple grayscale/edge statistics are fundamentally
limited descriptors for this task, not just poorly chosen. Confusion
matrix shows 6,908 of 10,584 actual destroyed buildings misclassified as
no-damage - a serious practical failure mode.
**Val vs test discrepancy:** macro F1 0.21 (val) vs 0.31 (test) - consistent
with the Phase 3 finding that val's destroyed class has only 403 instances,
making its per-class metrics unreliable. Test set is the trusted number.
**Status:** Confirmed. Establishes Baseline 2 floor - meaningfully better
than Baseline 1's noise-dominated binary output, but inadequate for the
real task. Motivates a learned deep model (Baseline 3 / Phase 5) that can
capture spatial/textural patterns simple statistics cannot.
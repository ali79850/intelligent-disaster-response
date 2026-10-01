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
## 2026-09 — Phase 4: Baseline 3 Compute Plan

**Decision:** Train Baseline 3 (simple CNN) CPU-only, locally. Defer Colab/
GPU setup to Phase 5, where the more sophisticated architecture will
actually need it.
**Reason:** Baseline 3 is deliberately small/simple; CPU training is
feasible within reasonable time given our measured ~7.4 min/epoch data
loading throughput from Phase 3, especially with a small epoch count.
**Status:** Confirmed.
## 2026-09 — Phase 4: SimpleCNN Architecture Verified

**Finding:** SimpleCNN (108,533 parameters, no skip connections) produces
correct output shape (2, 5, 1024, 1024) on a real batch through the
verified DataLoader. Ready for training.
**Status:** Confirmed.
## 2026-09 — Phase 4: Training Loop Smoke Test Passed

**Finding:** SimpleCNN trained on 5 tiles for 10 iterations shows clean,
monotonic loss decrease (1.8766 -> 1.1590), confirming model, loss
(class-weighted CrossEntropy, ignore_index=255), and optimizer are wired
correctly before committing to a full training run.
**Status:** Confirmed. Applying the lesson from the feature-extraction
hang: verify at small scale before scaling up.
## 2026-09 — Phase 4: SimpleCNN Performance Issue Found

**Finding:** Initial SimpleCNN architecture measured at ~4.2s/batch during
real training (confirmed via two consistent checkpoints: 419.3s/100 batches,
843.1s/200 batches), extrapolating to ~62 min/epoch - far too slow for the
"deliberately simple baseline" this is meant to be, and risks another
multi-hour unattended run.
**Cause:** Full 1024x1024 resolution convolutions in the first encoder
block, bottleneck output, and final decoder block are computationally
expensive even with small channel counts (16-64), since spatial size was
not reduced early enough.
**Fix:** Add a strided/pooled downsampling step before the first conv
block so most computation happens on smaller feature maps (e.g., downsample
to 256x256 or lower before the main conv stack), only upsampling back to
full resolution at the very end.
**Status:** In progress - re-verifying correctness and speed after fix.
## 2026-09 — Phase 4: SimpleCNN Revised Architecture Verified

**Finding:** Revised SimpleCNN (stride-4 stem downsampling to 256x256
before the main conv stack, single upsample back to 1024x1024 at the end)
verified correct: output shape (2,5,1024,1024) unchanged, parameter count
91,285 (down from 108,533), and smoke test shows clean monotonic loss
decrease (1.5507 -> 1.0568 over 10 iterations on 5 tiles).
**Status:** Confirmed correct. Proceeding to re-measure training speed.
## 2026-09 — Phase 4: Baseline 3 Training Complete

**Finding:** SimpleCNN trained for 3 epochs on full train set (1782 tiles).
Loss: epoch 1 = 1.0500 (24.0 min), epoch 2 = 0.9430 (23.4 min), epoch 3 =
0.8896 (16.5 min, unexplained speedup - not investigated further, not
concerning for a baseline). Consistent decreasing trend with diminishing
returns, as expected. All three epoch checkpoints saved to models/.
**Status:** Confirmed. Proceeding to evaluation on val/test.
## 2026-09 — Phase 4: Evaluation Script Memory Bug

**Finding:** evaluate_baseline3.py crashed with ArrayMemoryError trying to
concatenate all ~461 million pixels from the val set into single arrays
before computing metrics (~7GB simultaneous memory for predictions+targets).
**Fix:** Accumulate a running confusion matrix per batch instead of storing
raw pixel arrays - constant memory regardless of dataset size, standard
practice for large-scale segmentation evaluation.
**Status:** Fixing now.
## 2026-09 — Phase 4: Baseline 3 (Simple CNN) — Results

**Method:** SimpleCNN (91,285 params, stride-4 stem + pooling to 64x64
bottleneck, no skip connections), trained 3 epochs, class-weighted
CrossEntropy loss, ignore_index=255.
**Results (test set):** Macro F1 0.2719. Per-class F1: background 0.9450,
no-damage 0.4144, minor-damage 0.0000, major-damage 0.0000, destroyed
0.0000. Confusion matrix confirms the model NEVER predicts minor/major/
destroyed at all - complete collapse to a 2-class (background vs no-damage)
output despite class-weighted loss.
**Interpretation:** Comparable macro F1 to Baseline 2 (0.31) but the
failure mode is more severe - Baseline 2 at least partially learned
destroyed/minor classes. Hypothesis (not yet confirmed): the aggressive
16x spatial downsampling (1024->64 at bottleneck) with no skip connections
likely destroys fine spatial detail needed to detect small, localized
damage signals, which the earlier polygon analysis (Phase 4 Baseline 2
prep) showed can be as small as 141-292 pixels. This directly motivates
Phase 5's use of a real U-Net-style architecture WITH skip connections,
rather than being an incidental result - a baseline without skip
connections completely failing on minority classes is exactly the kind of
evidence that justifies the more sophisticated architecture, rather than
choosing it for complexity's sake.
**Status:** Confirmed, reported honestly despite poor performance.
## 2026-09 — Phase 5: U-Net Architecture Implemented and Verified

**Decision:** UNetResNet18 - ResNet-18 encoder (ImageNet-pretrained),
U-Net-style decoder with skip connections at 4 resolution levels, 6-channel
input (concatenated pre+post images, first conv layer modified with
duplicated pretrained weights across the extra channels).
**Reason:** Directly addresses Baseline 3's diagnosed failure (Phase 4) -
severe downsampling with no skip connections caused complete collapse on
minority damage classes. Skip connections let fine spatial detail bypass
the bottleneck.
**Pretrained weights:** ResNet-18 ImageNet1K_V1 weights, source: torchvision.
Used for general low/mid-level visual feature transfer (edges, textures)
from natural images - NOT domain-specific satellite imagery knowledge.
Documented per working rule L (responsible pretrained weight use).
**Verification:** Output shape (2,5,1024,1024) confirmed correct on real
batch. Total parameters: 14,344,741 (vs Baseline 3's 91,285).
**Status:** Confirmed correct. Proceeding to loss function design and
smoke test before any full training.## 2026-09 — Phase 5: Combined Loss (Weighted CE + Dice) Verified

**Finding:** CombinedLoss (weighted CrossEntropy + Dice, ignore_index=255)
verified: random predictions give loss=2.6948, near-perfect predictions
give loss=0.0004 - confirms the loss function mathematically rewards
correct predictions and properly excludes ignored pixels. Backward pass
succeeds with no NaN gradients.
**Status:** Confirmed. Ready for smoke test on the real U-Net model.
## 2026-09 — Phase 5: Local CPU Training Confirmed Infeasible for U-Net

**Finding:** Smoke test measured ~8.2s/sample average for UNetResNet18
(14.3M params) with combined loss, on CPU. Extrapolated to batch_size=2,
891 batches/epoch: ~4 hours/epoch, 12+ hours for even 3 epochs.
**Decision:** Move to Google Colab (free GPU tier) for all U-Net training,
per the Phase 0 hardware plan ("training can use Google Colab/Kaggle GPU
if necessary"). Local CPU remains used for correctness verification via
smoke tests only (as just demonstrated), never for full training runs on
this architecture.
**Status:** Confirmed. Proceeding to prepare Colab-compatible training
setup.
## 2026-09 — Phase 5: Colab Data Strategy

**Decision:** Colab notebook downloads the xBD Challenge training set
directly from xview2.org (same curl command used locally) and regenerates
masks/splits using our already-verified scripts, rather than uploading a
pre-processed package from local storage to Google Drive.
**Reason:** Colab has much better bandwidth than a typical home upload
connection, avoiding the slow-upload bottleneck (8.28GB package). Also
more reproducible - the notebook demonstrates the same documented,
verified pipeline (download -> generate masks -> generate splits) rather
than depending on a private Drive link only the author has access to.
**Trade-off:** Mask generation (a few minutes, not hours - verified in
Phase 3) must re-run each fresh Colab session, since only Google Drive
persists between sessions, not Colab's local disk.
**Status:** Confirmed.
## 2026-09 — Phase 5: Colab GPU Batch Size Selection

**Finding:** Empirical test on Colab T4 (15.64GB VRAM): batch_size=4 uses
8.52GB peak memory, batch_size=8 uses 14.42GB (92% of total, too tight for
a safe long run), batch_size=16 causes OutOfMemoryError.
**Decision:** Use batch_size=4 for full U-Net training - comfortable
headroom rather than the riskier batch_size=8, given the cost of a crash
partway through a multi-hour run.
**Status:** Confirmed.
## 2026-09 — Phase 5: Colab Training — Infrastructure Lessons

**What happened:** First full U-Net training run (5 epochs, ~35 min total)
completed successfully but the Colab runtime disconnected/reset before
checkpoints could be backed up, losing all work. Root causes: (1) Google
Drive mounting failed repeatedly with "credential propagation unsuccessful"
- a known Colab auth glitch, (2) a duplicated git clone caused a nested
directory path bug, (3) the runtime reset entirely, wiping /content/.
**Fix applied:** Rebuilt in a fresh notebook, verified each cell's output
before proceeding (rather than running several cells blind), and changed
the training loop to call files.download() after EVERY epoch rather than
only at the end - bounding potential data loss to a single epoch (~7 min)
instead of the full run.
**Result:** Second full training run completed cleanly, all 5 checkpoints
saved locally. Final loss 1.5320 (vs. 1.4651 in the lost run - close, not
identical, expected given no fixed random seed for this training script).
**Lesson:** For any future long-running Colab session, checkpoint and
persist immediately after every meaningful unit of work, never rely on
end-of-run saves, and don't trust Drive mounting to work on the first
attempt - always have a working fallback (direct browser download) ready
before starting a real run, not improvised after a failure.
## 2026-09 — Phase 5: U-Net (5 epochs) — Results

**Method:** UNetResNet18, 5 epochs, combined weighted CE + Dice loss,
batch_size=4, Adam lr=1e-4, trained on Colab T4 GPU.
**Results (test set):** Macro F1 0.2862 - below Baseline 2 (0.31), modestly
above Baseline 3 (0.27). Per-class F1: background 0.9584, no-damage 0.1208,
minor-damage 0.0345, major-damage 0.0512, destroyed 0.2664.
**Interpretation:** Unlike Baseline 3, this model does NOT collapse - it
achieves high recall on destroyed (0.8080) and minor-damage (0.2132),
proving skip connections do let the model detect small/localized damage
signals that Baseline 3 completely missed. However, precision is poor
across all damage classes (destroyed precision 0.1595, minor-damage
precision 0.0187) - the model over-flags damage broadly and confuses
classes, rather than being cleanly confident. Net effect on macro F1 is
currently a wash against Baseline 2's simpler approach, but the underlying
behavior (high recall, low precision) is a more tractable problem than
Baseline 3's complete failure to detect anything.
**Not yet conclusive:** Only 5 of a likely-larger needed epoch count.
Training loss was still decreasing meaningfully at epoch 5 (delta -0.0864,
epoch 4->5), suggesting the model has not converged. This result should be
treated as an interim checkpoint, not a final verdict on the architecture.
**Status:** Confirmed, reported honestly despite not yet beating Baseline 2.
Continued training planned before drawing final conclusions.
## 2026-09 — Phase 5: U-Net (10 epochs) — Results, Beats Baseline 2

**Method:** UNetResNet18, resumed training from epoch 5 checkpoint for 5
additional epochs (10 total), same combined weighted CE + Dice loss,
batch_size=4, Adam lr=1e-4.
**Results (test set):** Macro F1 0.3380 - up from 0.2862 at epoch 5, and
now exceeds Baseline 2's 0.31 for the first time. Per-class F1: background
0.9557, no-damage 0.3921 (up substantially from 0.12 at epoch 5),
minor-damage 0.0266, major-damage 0.0340, destroyed 0.2815 (recall 0.7570,
precision still weak at 0.1729).
**Interpretation:** Additional training meaningfully improved no-damage
and destroyed detection. Minor/major-damage remain weak - these are likely
the hardest classes to distinguish (subtle visual differences between
"some damage" and "moderate damage"), not simply solved by more epochs at
the current learning rate/architecture. Precision on destroyed remains the
core weakness - the model still over-flags broadly rather than being
confident and precise.
**Comparison across all approaches (test macro F1):** U-Net 10ep 0.3380 >
Baseline 2 (RF) 0.31 > U-Net 5ep 0.2862 > Baseline 3 (CNN, no skip) 0.27.
**Val vs test discrepancy:** macro F1 0.2808 (val) vs 0.3380 (test) -
consistent with the known thin destroyed-class sample in val (403
instances, documented since Phase 3). Test remains the trusted comparison
point.
**Status:** Confirmed, real improvement demonstrated with honest per-class
breakdown.
## 2026-09 — Phase 6: Error Analysis — Concrete Failure Cases Identified

**Finding:** Visualized specific test tiles confirm the aggregate confusion
matrix pattern concretely. hurricane-matthew_00000000: GT has 25,731
minor-damage pixels, model predicted only 79 (near-total miss), while
over-predicting destroyed (545 GT vs 61,130 predicted). palu-tsunami_
00000001: GT has 1,213 major-damage + 1,243 destroyed pixels (small,
localized), model predicted 61,859 + 86,423 respectively (~50-70x
over-prediction).
**Status:** Confirmed via direct tile inspection, not just aggregate
metrics. Visual inspection of colorized overlay pending to determine
whether over-prediction is spatially concentrated on real buildings
(severity-confusion) or scattered (localization failure).
**Visual confirmation:** hurricane-matthew_00000000 overlay shows the model
correctly localizes the exact same building cluster as ground truth (error
map shows a clean building-shaped silhouette, not scattered noise) but
misclassifies the severity - predicting destroyed (red) where ground truth
says minor-damage (yellow). This confirms the failure is SEVERITY
DISCRIMINATION, not spatial localization - the model has learned "where
damage is" correctly but is biased toward the most extreme class once
triggered, rather than calibrating to actual damage degree. This is a
more tractable, specific problem than Baseline 3's complete failure to
localize anything at all.
## 2026-09 — Phase 7: Grad-CAM Verified

**Finding:** SegmentationGradCAM (adapted for dense per-pixel output,
region-restricted backprop) verified: correct shape (1024,1024), values
in [0,1], sensible non-degenerate mean (0.0399 - concentrated activation,
not uniform or all-zero).
**Status:** Confirmed. Proceeding to targeted explanation of the specific
wrong "destroyed" prediction identified in Phase 6 error analysis.
## 2026-09 — Phase 7: Grad-CAM Confirms Correct Localization, Severity Miscalibration

**Finding:** Targeted Grad-CAM for the wrong "destroyed" prediction on
hurricane-matthew_00000000 (60,913 pixels in region of interest) shows the
highlighted activation sits directly on the actual building cluster, not
on unrelated image regions (fields, river, or other artifacts, aside from
a faint response on the known black no-data border).
**Interpretation:** Combined with Phase 6's error analysis, this confirms
a coherent two-part diagnosis: the model correctly attends to real
buildings when making damage predictions (localization is genuinely
learned, not a shortcut), but is miscalibrated on severity - the internal
representation that triggers "destroyed" appears to activate on the same
buildings that are actually only minor-damage, suggesting the decision
boundary between damage severity levels is not yet well-separated in the
learned feature space, rather than the model attending to the wrong
things entirely.
**Limitation restated:** this Grad-CAM explanation targets one intermediate
layer (enc4 bottleneck) and does not capture the decoder/skip-connection
pathway's contribution to the final decision - it shows where gradient
sensitivity concentrated at that layer, not a complete causal account.
**Status:** Confirmed. This is now a well-supported, specific finding for
the project's error analysis and limitations sections - not a vague
"the model needs more training" statement, but a concrete hypothesis
(severity miscalibration despite correct localization) backed by both
quantitative (confusion matrix) and qualitative (visual, Grad-CAM)
evidence from three independent analysis angles.
## 2026-09 — Phase 7/8 Transition: Model Improvement Deferred as Future Work

**Decision:** Move to Phase 9 (inference system) rather than continue
model training/tuning at this point.
**Considered but deferred:**
1. Targeted experiment: reduce destroyed-class weight or dice_weight to
   test whether current class weighting overcorrects and causes the
   observed destroyed-over-prediction bias (Phase 6/7 finding).
2. Incorporate xBD tier3 (~17GB supplemental data, same schema/annotation
   methodology as train, zero cross-dataset reconciliation risk) for
   greater disaster-event diversity in training.
3. A second dataset (e.g., RescueNet) for cross-dataset generalization -
   explicitly rejected as a near-term action due to schema/annotation
   reconciliation risk and effort; remains a valid long-term extension.
**Reason for deferring all three:** the project currently has a complete,
honestly-evaluated model with a well-documented limitation, but zero
inference/API/dashboard infrastructure. Per the original project scope,
demonstrating the full pipeline (training -> inference -> API -> dashboard)
is a larger gap than incremental model improvement at this stage.
**Status:** Confirmed. Items 1 and 2 are prioritized future work if time
allows after Phase 9-10 are complete; item 3 is longer-term future work.
## 2026-09 — Phase 9: Inference Engine Implemented and Verified

**Finding:** DamageInferenceEngine (src/inference/engine.py) verified
against hurricane-matthew_00000000 - destroyed pixel count (61,130)
matches Phase 6 error analysis exactly, confirming no preprocessing/
normalization drift between training-time (XBDDataset) and inference-time
logic.
**Observation:** Mean confidence for this tile is 0.9552 (high) despite
the model being confidently WRONG about severity here (Phase 6/7 finding).
This confirms overall mean confidence is an insufficient uncertainty
signal on its own - per-class or per-region confidence breakdowns would
better surface exactly the miscalibration already diagnosed. Noted as a
consideration for the eventual dashboard's confidence/review-flagging
feature (per original spec section 12/13), not yet implemented.
**Status:** Confirmed. Inference engine ready to back an API layer.
## 2026-09 — Phase 9: Report Generation Verified — Refined Diagnosis

**Finding:** generate_report() on hurricane-matthew_00000000 shows
per-class confidence: background 0.9656, no-damage 0.3927, minor-damage
0.2506, major-damage 0.4772, destroyed 0.8243. This refines the Phase 6/7
diagnosis with precision: the model is NOT uniformly uncertain about
severity - it is specifically, measurably low-confidence on the three
MIDDLE severity classes (no-damage, minor, major), while being confidently
(if sometimes wrongly) certain about the extremes (background, destroyed).
requires_human_review correctly triggers True, flagging exactly these
three classes.
**Impact:** This is a more precise, actionable characterization of the
model's weakness than "severity miscalibration" alone - it specifically
struggles to distinguish gradations of damage, while reliably
distinguishing "clearly nothing happened" from "something happened here."
Valuable for the project's final limitations section and for any future
targeted improvement work (e.g., the deferred class-weighting experiment
could specifically target improving middle-class calibration).
**Status:** Confirmed. Report generation ready for API integration.
## 2026-09 — Phase 8: Geospatial Metadata Verified

**Finding:** Verified hurricane-matthew_00000000's building polygons
(features.lng_lat) carry genuine real-world coordinates: longitude
-73.740 to -73.737, latitude 18.196 to 18.198 - confirmed as coastal
Haiti, consistent with Hurricane Matthew's actual October 2016 landfall
location and the tile's capture_date metadata (2016-10-09, the exact
landfall date). This is real georeferencing, not synthetic/relative
coordinates.
**Impact:** Phase 8 can legitimately build genuine map-based geospatial
visualizations (real lat/lon on an actual map, e.g. via Folium), not a
simplified placeholder. Each building's exact real-world location is
available, enabling honest damage heatmaps and spatial distribution
analysis.
**Also confirmed:** metadata includes gsd (2.77 m/pixel here, consistent
with Phase 2's documented range), sensor info, off_nadir_angle, sun
azimuth/elevation - these support future photometric/viewing-angle
analysis if needed, but are not required for core Phase 8 scope.
**Status:** Confirmed. Proceeding to build tile-level bounding box
extraction and map visualization.
## 2026-09 — Phase 8: Damage Map Implemented and Verified

**Finding:** build_damage_map() generates a real Folium map from building
centroids, verified against hurricane-matthew_00000000: 105 buildings
plotted (matching Phase 4's polygon utility count exactly), class
breakdown (minor-damage 97, no-damage 3, destroyed 3, major-damage 2)
consistent with all prior analysis of this tile (Phase 3 visual overlay,
Phase 6 error analysis).
**Status:** Confirmed pending visual browser verification.
## 2026-09 — Phase 10: FastAPI Backend Implemented

**Finding:** app/backend/main.py starts cleanly, loads the model once at
startup (unet_epoch10 on cpu), confirmed via startup log. Fixed a relative-
path bug where DamageInferenceEngine's default config path assumed being
run from the project root; resolved using Path(__file__).resolve() to
build absolute paths regardless of the working directory the server is
launched from.
**Status:** Confirmed server starts correctly. Proceeding to test actual
endpoints.
## 2026-09 — Phase 10: API Verified End-to-End

**Finding:** POST /api/analyze tested via curl with real image files
(hurricane-matthew_00000000 pre/post). Response matches every previously
verified value exactly: destroyed 61,130 pixels, affected_area_percentage
6.22%, all per-class confidence values, requires_human_review flag and
flagged classes. Confirms the full stack (HTTP -> file upload -> inference
engine -> report generation -> JSON response) works correctly with zero
drift from script-based verification.
**Status:** Confirmed. Core API endpoints (/api/health, /api/model-info,
/api/analyze) all verified working.
## 2026-09 — Phase 10: Frontend Scaffolded

**Decision:** React + TypeScript + Vite + Tailwind CSS, per original spec.
Scaffolded via `npm create vite@latest . -- --template react-ts`, Tailwind
installed via the Vite plugin (@tailwindcss/vite), verified working via a
temporary test class before building real components.
**Status:** Confirmed. Proceeding to build the actual upload/analyze/
results dashboard, scoped to exactly what the API currently supports
(upload, analyze, structured report display) - not the full dashboard
vision from the original spec, which includes features (map view, review
workflow, PDF export) not yet built on the backend.
## 2026-09 — Phase 10: Investigated Apparent Discrepancy — Confirmed Correct

**Finding:** User observed santa-rosa-wildfire_00000137 showing "No Damage"
at 0% with a non-null confidence (0.27), appearing inconsistent.
Investigation via direct script call confirmed this is NOT a bug: the
tile genuinely has 3 no-damage pixels (out of 1,048,576), which rounds to
0.00% at 2 decimal places but is not truly zero, so the confidence
guard-clause (mask.sum() > 0) correctly computes a real value. Minor/major
-damage, which are genuinely 0 pixels, correctly show n/a.
**Impact:** No backend bug. UI display precision (2 decimals) can make
very small non-zero pixel counts appear misleadingly as "0%" alongside a
real confidence value - a legitimate clarity issue for the frontend to
address (e.g., showing "<0.01%" instead of "0.00%" for tiny non-zero
counts), not a data correctness issue.
**Status:** Confirmed correct behavior; UI precision improvement noted.
## 2026-09 — Phase 10: New Error Pattern Discovered — Vegetation/Burn Confusion

**Finding:** User visually identified apparent fire damage in
santa-rosa-wildfire_00000137's post-disaster image. Ground truth confirms
only 2 buildings in this tile, both destroyed (1,412 ground-truth destroyed
pixels). Model predicted 5,063 destroyed pixels, but only 53 pixels (3.7%)
overlap with the real destroyed buildings - the model missed 96% of the
actual destroyed-building pixels and instead flagged ~5,010 pixels
elsewhere in the tile, almost certainly the visually similar burned/
discolored vegetation area.
**Interpretation:** This is a genuinely different, newly discovered failure
mode from the Phase 6/7 finding (severity confusion among correctly-
localized buildings). Here, the model appears to react to general visual
damage/burn-scarring signals (which resemble building destruction
texturally) rather than specifically building-level damage, and misses
the actual small building footprints in a landscape dominated by fire-
affected vegetation. This is conceptually similar to Baseline 1's
weakness (confusing general visual change with actual damage), suggesting
the U-Net has not fully escaped this failure mode for fire-disaster tiles
with sparse building coverage and dominant vegetation change.
**Significance:** This was found through genuine human review of a model
result via the dashboard - exactly the human-in-the-loop review workflow
this project's design anticipates. Credit: identified by direct visual
inspection of the deployed interface, not by a pre-planned test case.
**Status:** Confirmed via direct pixel-level ground truth comparison.
Documented as a real, additional limitation - candidate for future work
(e.g., additional training on fire-type disasters specifically, or
explicit vegetation-masking preprocessing).
## 2026-09 — Phase 11: LLM Summary Layer Implemented and Verified

**Decision:** Use Groq's free-tier API (openai/gpt-oss-20b) for narrative
summary generation, not Anthropic's API, per user preference for a
no-cost option. Verified against Groq's actual account model list via
direct API call, not documentation alone, after two incorrect model-name
guesses failed with 404 errors.
**Bug found and fixed:** Initial implementation returned the model's raw
internal reasoning trace instead of the final answer, because a fallback
(`content or reasoning`) silently substituted unintended text when
`content` came back empty - caused by gpt-oss's default reasoning
behavior consuming the entire token budget before producing a final
answer. Fixed by setting `reasoning_format: "hidden"` and
`reasoning_effort: "low"` via the Groq API, and removing the dangerous
silent fallback in favor of an explicit error if content is ever empty.
**Verification:** Tested against hurricane-matthew_00000000's known
report. Generated summary correctly cites all real numbers (6.22%
affected, 5.83% destroyed, etc.), correctly explains the human-review
flag with specific low-confidence classes, includes the required
disclaimer, and contains zero fabricated content (casualties, population,
infrastructure) per an automated forbidden-term check.
**Status:** Confirmed. Ready for API integration.
## 2026-09 — Phase 11: /api/summarize Endpoint Verified

**Finding:** POST /api/summarize tested via real HTTP request using the
actual hurricane-matthew_00000000 report. Status 200, summary content
matches the direct script-based test exactly in accuracy and structure.
Confirms the full stack (FastAPI -> summarizer module -> Groq API ->
response) works correctly end-to-end.
**Design note:** Implemented as a separate endpoint from /api/analyze
(not bundled automatically) so the core ML damage assessment never fails
or slows down due to the LLM call, an external dependency with its own
failure modes, being optional and separable per the original spec's
framing of the LLM layer as an enhancement, not a requirement.
**Status:** Confirmed. Ready for frontend integration.
## 2026-09 — Phase 12: Unit Tests for Metrics and Report Logic

**Finding:** 9 tests covering segmentation_metrics.py (precision/recall/
IoU/F1 math, ignore-index handling, zero-division safety) and report.py
(review-flag triggering, per-class confidence null-handling, affected-area
calculation, limitations always present). All use hand-calculated expected
values, not placeholder assertions - e.g. test_known_partial_overlap
verifies precision=2/3 against a manually traced confusion matrix.
**Status:** Confirmed, 9/9 passing. Proceeding to inference engine and
API-level tests.
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
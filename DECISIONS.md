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
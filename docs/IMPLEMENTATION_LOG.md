# Implementation Log

## 2026-04-19

- Initialized project structure and Python package scaffolding.
- Added dependency management via `pyproject.toml` (uv compatible).
- Created execution plan file mapped from the provided `plan.md`.
- Started building modules for:
  - dataset acquisition
  - preprocessing
  - segmentation
  - feature matching
  - 3D reconstruction
  - evaluation and visualization
- Implemented full package modules in `src/cvfinal/` covering Steps 1-10.
- Added `run_full_pipeline.py` master orchestration script with CLI flags.
- Added dataset and utility scripts under `scripts/`.
- Added `docs/RESOURCE_LOG.md` to track datasets, techniques, and references used.
- Performed syntax validation with `uv run python -m compileall src run_full_pipeline.py scripts`.
- Patched incremental SfM fallback to preserve pose continuity when `solvePnPRansac` fails.
- Updated dataset downloader to use working COIL-100 processed URL and Objectron index-based video retrieval.
- Verified dataset acquisition end-to-end (`scripts/download_datasets.py`).
- Verified full pipeline run on Objectron sample video (all stages completed).
- Verified full pipeline run on COIL ordered sequence (all stages completed after reconstruction robustness patch).
- Added directory reset safety to avoid mixing artifacts across runs.
- Added `docs/RUN_RESULTS.md` with reproducible commands and observed outputs.
- Added `scripts/generate_report_draft.py` and generated `docs/PROJECT_REPORT_DRAFT.md` from pipeline metrics.
- Refactored Step 3 matching to carry per-frame feature indices for stable 2D-3D track propagation.
- Rebuilt Step 4/5 incremental SfM to use feature-index tracks, robust triangulation filtering (depth + reprojection), and stronger PnP-RANSAC pose updates.
- Removed interactive ROI dependency from the default pipeline route and kept automatic per-frame mask generation active.
- Validated pipeline on `data/raw/your_video.mp4` with improved metrics (`num_points_3d=374`, `mean_reprojection_error=3.21`).

This log will be appended as implementation progresses.

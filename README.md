# Multi-Technique 3D Reconstruction from Rotating Video

This repository implements a full classical Computer Vision pipeline aligned with the project plan in `plan.md`.

## What is implemented

- Frame extraction from video
- Preprocessing (CLAHE, Canny, overlays)
- Object mask generation (GrabCut + morphology + contour filtering)
- SIFT feature extraction and pairwise matching with Lowe ratio test
- Two-view pose recovery and triangulation
- Incremental reconstruction using PnP + triangulation
- Dense optical flow visualization (toggle)
- PCA alignment of reconstructed point cloud
- BoVW + SVM object classification with confusion matrix
- Quantitative output metrics and report-ready artifacts

## Project structure

- `run_full_pipeline.py`: master script for full execution
- `scripts/download_datasets.py`: dataset download and preparation (Objectron + COIL-100)
- `scripts/extract_frames.py`: standalone frame extraction utility
- `scripts/prepare_coil_sequence.py`: generate ordered frame sequence from COIL for reconstruction verification
- `scripts/generate_report_draft.py`: auto-generate report draft markdown from metrics
- `src/cvfinal/`: pipeline modules
- `docs/EXECUTION_PLAN.md`: implementation plan
- `docs/IMPLEMENTATION_LOG.md`: progress log
- `docs/RESOURCE_LOG.md`: datasets/references/techniques log

## Setup (uv)

```bash
uv sync
```

## Download and prepare datasets

```bash
uv run python scripts/download_datasets.py --data-root data --coil-classes 1 2 3 4
```

## Option A: Run on Objectron sample video

```bash
uv run python run_full_pipeline.py \
  --video data/raw/objectron_sample.MOV \
  --classification-root data/interim/classification \
  --enable-optical-flow \
  --frame-step 8 \
  --reproj-threshold 15
```

If segmentation is unstable for a scene, disable masking to inspect geometric behavior first:

```bash
uv run python run_full_pipeline.py \
  --video data/raw/objectron_sample.MOV \
  --classification-root data/interim/classification \
  --disable-mask \
  --frame-step 8 \
  --reproj-threshold 15
```

If the mask cuts small object parts (for example bottle caps), expand the foreground mask:

```bash
uv run python run_full_pipeline.py \
  --video data/raw/objectron_sample.MOV \
  --classification-root data/interim/classification \
  --interactive-roi \
  --mask-dilate 4 \
  --frame-step 8
```

Note: segmentation masks are generated per frame (not reused from the first frame), which helps keep foreground features on moving/rotating objects.

## Option B: Run reconstruction verification on COIL sequence

```bash
uv run python scripts/prepare_coil_sequence.py \
  --class-dir data/interim/classification/obj1 \
  --out data/interim/coil_sequence \
  --step 4

uv run python run_full_pipeline.py \
  --frames-dir data/interim/coil_sequence \
  --classification-root data/interim/classification \
  --enable-optical-flow
```

## Outputs

All generated artifacts are written under `data/output/`, including:

- `preprocess_overlays/`
- `segmentation/`
- `matches/`
- `optical_flow/` (if enabled)
- `reconstruction/` (`.ply`, plots, GIF, overlay)
- `classification/` (`classification_report.txt`, confusion matrix)
- `metrics.json`

## Generate report draft

```bash
uv run python scripts/generate_report_draft.py \
  --metrics data/output/metrics.json \
  --out docs/PROJECT_REPORT_DRAFT.md
```

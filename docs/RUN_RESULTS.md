# Run Results

## Verified Commands

1. Download datasets

```bash
uv run python scripts/download_datasets.py --data-root data --coil-classes 1 2
```

Status: success

2. Full pipeline on Objectron sample video

```bash
uv run python run_full_pipeline.py \
  --video data/raw/objectron_sample.MOV \
  --classification-root data/interim/classification \
  --enable-optical-flow \
  --frame-step 20
```

Status: success

Current `data/output/metrics.json` snapshot:

- `num_points_3d`: 5
- `mean_reprojection_error`: 3988.9026
- `classification_accuracy`: 1.0

3. COIL ordered frame sequence reconstruction verification

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

Status: success

## Notes

- The pipeline now resets intermediate/output directories at each run to avoid cross-run contamination.
- Objectron reconstruction currently yields sparse points with high reprojection error under the no-calibration assumption (`f = image width`).
- COIL sequence is included as an additional real dataset verification path and can be reused as a pre-check before custom video runs.
- For challenging scenes, use `--frame-step 8 --disable-mask` to retain enough correspondences.
- Reconstruction now preserves raw cloud if strict reprojection filtering would collapse the output to very few points.

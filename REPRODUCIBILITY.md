# Reproducibility

## Environment (this run)
- Windows 11, Python via `.venv` at repo root
- Key packages (from `pip show`): `segmentation-models-pytorch==0.5.0`, `rasterio==1.5.1`, `geopandas==1.1.4`
- CPU only (no CUDA available this session)
- Commit this work is built on: `396d4a1` (main, as of session start)

## Commands to reproduce fold 0

```bash
# sanity check: config loads, labels load, corrected small-object share reported
python scripts/check_pipeline.py

# corrected evaluation, legacy checkpoint (no retrain) -- fast, ~2 min/model
PYTHONPATH=src python -m landslide_pipeline.evaluate --fold 0 --model-key baseline
PYTHONPATH=src python -m landslide_pipeline.evaluate --fold 0 --model-key comparison

# corrected retrain from scratch (slow on CPU -- hours, not minutes, for the full fold)
PYTHONPATH=src python -m landslide_pipeline.train --fold 0 --model-key baseline
PYTHONPATH=src python -m landslide_pipeline.train --fold 0 --model-key comparison
# then re-run evaluate.py -- it prefers experiments/corrected_experiment_v1/checkpoints/
# over the legacy Trained Model/ checkpoints if both exist.

# remaining folds (not run this session -- each is the same two commands with --fold 1..7)
```

## Seeds
`spatial_cv.yaml: random_seed: 42` is the single seed used for K-Means clustering, Python's
`random`, NumPy, and PyTorch (`training.py::seed_everything`). No multi-seed reruns were performed
this session (Issue 24 notes this is addressed at the single-seed level; multi-seed variance
estimation is a separate, larger undertaking not attempted here).

## Data manifest (spot-checked, not machine-generated)
- Labels: `data/raw/labels/Lanslides_Ditwah_2025.shp` — 4,225 polygons, EPSG:32644 after reprojection.
- Sentinel-2: `raw/sentinel2/tile{0,1,2,3}/openEO_2025-{10-31,11-02,11-20}Z.tif` — all three dates
  are confirmed < the 2025-11-28 pre-event cutoff by direct filename inspection.
- DEM: `raw/dem/dem_tile{0,1,2,3}.tif`, Copernicus GLO-30 resampled to 10 m.
- No separate per-sample provenance manifest file was generated this session (Issue 18, deferred).

## What "corrected" currently means, precisely
A result produced by `src/landslide_pipeline/evaluate.py` (this session's new module) includes:
cloud/shadow/nodata-masked metrics, PR-AUC and ROC-AUC, and labels/small-object split computed
from the corrected minor-axis formula. It does **not** yet include: retraining with the masked
loss (unless `train.py` was explicitly run for that fold/model — check for a checkpoint under
`experiments/corrected_experiment_v1/checkpoints/` vs. a fallback to `Trained Model/`),
slope-stratified negative sampling, a validation-selected threshold, or more than one spatial fold.
Every results table in `EXPERIMENTAL_RESULTS.md` states explicitly which checkpoint (legacy vs.
corrected-retrained) and which fold produced it.

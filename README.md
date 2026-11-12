# Landslide Forecasting System

A research prototype for pre-event landslide susceptibility mapping in Sri Lanka using Sentinel-2 imagery, DEM-derived terrain features, and a U-Net / DeepLabV3+ segmentation pipeline.

## Scientific status

This repository is best viewed as a scientifically honest prototype and pipeline scaffold, not a final publication-grade experimental result set.

The core ideas are in place:
- a 16-channel input representation combining spectral bands with vegetation and terrain indices,
- 64 x 64 patch generation at 10 m resolution,
- a spatially grouped evaluation strategy,
- model comparison between U-Net and DeepLabV3+,
- a synthetic demo that exercises the trained checkpoints without live satellite fetches.

What is not yet present in the checked-in repository is the completed real-data experiment: full training logs, reproducible fold outputs, final metrics, ablation tables, and a validated checkpoint tied to a dataset manifest.

## Research scope and limitations

- This project models landslide susceptibility, not real-time forecasting or operational early warning.
- Inputs must remain pre-event with respect to the landslide inventory cutoff.
- A 30 m minimum mappable unit introduces a known small-object limitation.
- Cloud/shadow/no-data pixels must be masked and excluded; they are not valid negatives.
- Spatial validation must remain geographically separated and cannot be replaced by a random split.

## Data and service credit

- Landslide inventory: ACCIMT Cyclone Ditwah post-event mapping for Sri Lanka.
- Satellite data: Copernicus Sentinel-2 Level-2A and Copernicus DEM GLO-30 via the Copernicus ecosystem.
- This repository is for research and planning use only and does not establish operational land-slope or early-warning risk predictions.

## Repository files

- `notebooks/landslide_pipeline.ipynb` — main pipeline notebook
- `configs/` — acquisition, preprocessing, model, and spatial-CV settings
- `docs/architecture.md` — design rationale and model notes
- `docs/research_audit.md` — repository audit and scientific risk review
- `docs/reproducibility.md` — reproducibility and research-readiness guidance
- `web/` — synthetic demo server for model interaction without live geo-data
- `data/` — local processed/raw data, not redistributed in the repository
- `Trained Model/` — trained checkpoints when available locally

## Current implementation state

### Implemented

- research framing and method design,
- model and config scaffolding,
- synthetic demo pipeline,
- notebook-based workflow and technical notes,
- non-destructive repository documentation and audit trail.

### Not yet complete in this repo

- final real-data training run,
- full eight-fold spatial validation results,
- fold-level confusion matrices and metric tables,
- full ablation study,
- reproducible experiment manifest,
- publication-ready figures.

## Setup

### Local Python environment

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Web demo

```bash
pip install -r web/requirements.txt
python web/app.py
```

Open http://127.0.0.1:5000 after starting the server.

Important: the web demo requires a trained checkpoint under `Trained Model/`. If a checkpoint is absent, the app will start in a degraded state and report a clear error instead of crashing at import time.

## Pipeline utilities

A reusable Python package is now available under `src/landslide_pipeline/` for the research pipeline foundation:

```bash
PYTHONPATH=src python scripts/check_pipeline.py
```

This validates that the configuration loads and that the label shapefile is present before a real-data run.

## Documentation

- `docs/architecture.md` – design rationale and intended model architecture decisions
- `docs/research_audit.md` – technical audit of the repository and scientific risks
- `docs/reproducibility.md` – current reproducibility obligations and readiness note

## Reproducibility and validation expectations

This repository should not be treated as a validated research result until the following exist:

1. a full real-data acquisition and preprocessing record,
2. fixed spatial fold assignments,
3. training logs for all folds,
4. a manifest tying checkpoints to config and dataset versions,
5. per-fold metric tables and visual diagnostics,
6. a written statement of scientific limitations and remaining uncertainty.

## Quick note

The project is a promising prototype for landslide susceptibility modeling, but the checked-in repository still requires a real-data training and validation cycle before any final publication claim can be made.

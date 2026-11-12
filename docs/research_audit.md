# Technical audit report

## Executive summary

The repository contains a thoughtful research prototype for pre-event landslide susceptibility mapping based on Sentinel-2 plus DEM features. The underlying scientific design is sound at a high level: a segmentation pipeline, 16-channel patch inputs, a pre-event time constraint, spatially grouped validation, and a comparison between U-Net and DeepLabV3+.

However, the codebase is not yet a complete, fully validated research experiment. The project currently includes configuration scaffolding, a research notebook, a synthetic web demo, and placeholder documentation. It does not include a complete real-data training run, final eight-fold spatial validation results, publication-quality ablation tables, or a reproducible end-to-end training log chain tied to a checked-in checkpoint and dataset manifest.

This is a research prototype with a credible architecture, not a fully published experimental pipeline.

## Repository audit

### Implemented

- Core study framing and scientific scope are documented in the repository README and architecture notes.
- The workflow is structured around the pre-event landslide susceptibility problem using Sentinel-2 and DEM inputs.
- Configurations for acquisition, preprocessing, model training, and spatial cross-validation exist under `configs/`.
- The notebook `notebooks/landslide_pipeline.ipynb` captures the main pipeline flow: acquisition -> preprocessing -> sampling -> spatial validation -> modeling -> evaluation.
- A minimal web demo exists in `web/` to run a trained model on synthetic patches with a 16-channel feature representation.
- A final `README` explains the intended research scope and setup steps, including the requirement to use real data and credentials.

### Partially implemented

- The acquisition and preprocessing strategy is described in configuration files and notebook cells, but no ready-to-run full real-data processing script is checked in as a standalone production workflow.
- The spatial CV design is specified in YAML, but the actual fold assignments are tied to notebook execution and not yet captured as a reproducible, versioned artifact outside the notebook state.
- The model comparison and loss configuration are defined in YAML, but no final training metrics or checkpoint manifests are stored in the repo.
- The demo app does run on synthetic patches, but it is intentionally not a real-location prediction tool and depends on trained checkpoints being present under `Trained Model/`.

### Synthetic-only or smoke-test validated

- The web app runs only on procedurally generated synthetic patches and heuristic channel estimates from an uploaded RGB photo.
- The project description explicitly identifies the demo as a mechanism demo rather than a scientific validation.
- The repository includes no fully trained model evaluation artifacts on real data, so all performance metrics remain pending a real run.

### Real-data validated

- No end-to-end run with real Sentinel-2 imagery and a full ACCIMT landslide inventory is present in the checked-in repository state.
- No final eight-fold spatial validation statistics, confusion matrices, or segmentation metrics were committed from actual training.
- No real model checkpoint from a real-data experiment is included or traceably linked to a dataset version.

### Missing for complete scientific experiment

- A persistent dataset-manifest capture with exact acquisition dates, image IDs, cloud masks, and provenance metadata.
- A reproducible processing pipeline as scripts or project commands, not only notebook cells.
- Stored spatial fold assignments and the exact buffer distances used for each fold.
- Separate training/validation/test data definitions with no random leakage.
- Inner validation or hyperparameter tuning separated from final holdout spatial folds.
- Real training logs, checkpoint metadata, and evaluation outputs for all folds.
- A clear distinction between development smoke tests and final research experiments.
- Publication-quality figures, tables, and a finalized research readiness statement.

## Data leakage and scientific risk review

### Likely leakage or contamination risks

1. Random train/validation splits in notebook workflows would be inappropriate for geographically clustered landslide data if used as the primary evaluation. Spatial data should be split by clusters and buffered regions.
2. Any unbuffered patch extraction around positive polygons or nearby non-landslide negatives can cause overlapping training and test patches if the patch stride is too small relative to the buffer geometry.
3. Using imagery from a period that crosses the event cutoff without enforcing a strict pre-event filter would leak post-event information into the inputs.
4. If normalization statistics are computed over the full dataset rather than on the training folds only, the test fold information can influence the model.
5. If model selection or threshold tuning uses the final holdout fold, the reported performance becomes optimistic and is not a true test set estimate.

### Performance risks

- The sample imbalance between landslide and non-landslide pixels can make accuracy values misleading if confusion matrices and class-weighted metrics are not reported.
- A small-object cohort is explicitly identified as a separate challenge, and it is easy to dilute the true difficulty by excluding or down-weighting small landslides.
- Failing to exclude invalid pixels from both loss and evaluation can generate artificially stable metrics while silently treating cloud/shadow/no-data as genuine negatives.
- Reporting only the best fold or the best threshold can create a positively biased research narrative.

## Reproducibility risks

- The repository depends on real external satellite data and credentials; the required dataset cannot be reconstructed without those sources.
- The notebook is not a pure script; reproducibility depends on manual execution order and cell state.
- Model checkpoint names and training metadata are not yet versioned for all runs.
- There is no persistent experiment manifest linking config, dataset, model, and evaluation files.

## Changes made for repository reliability

During this pass, the project was stabilized in a conservative way without pretending that real scientific results were produced.

- The README was updated to explicitly describe the current status, scientific limitations, and the distinction between smoke tests and real-data research.
- The web demo was hardened so it fails clearly if model checkpoints are missing instead of crashing on import.
- The demo dependency list in `web/requirements.txt` was updated to include required runtime libraries for the model and preview pipeline.
- A dedicated audit document was added to record the repository state, scientific risks, and missing validation evidence.
- A reproducibility and research-readiness note was added to separate the current prototype from the final journal-grade experiment.

## Bottom line

The repository is a strong foundation for a geospatial deep-learning landslide susceptibility study, but the checked-in codebase is not yet ready to claim reproducible, real-data, publication-grade results.

The present repository should be treated as a scientifically honest prototype that is ready for the next phase of real-data acquisition, rigorous spatial validation, and full experimental logging.

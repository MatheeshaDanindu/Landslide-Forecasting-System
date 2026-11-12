# Reproducibility and research readiness notes

## Research objective

This project targets pre-event landslide susceptibility mapping using Sentinel-2 multispectral imagery and Copernicus DEM-derived terrain features. The prediction target is a pixel-level susceptibility probability map. The implementation is not intended to forecast landslide timing or produce an operational early-warning product.

## Scientific scope

The current code and configuration intentionally support the following constraints:

- only pre-event imagery is admissible for model input;
- no real-time rainfall or hydrology signal is included;
- landslide labels are derived from the ACCIMT post-event inventory;
- evaluation must be spatially grouped rather than randomly split;
- the separate small-object cohort remains a distinct analysis stream rather than being silently merged into the main training mask.

## Supported pipeline structure

The repository is designed around the following sequence:

1. acquire Sentinel-2 and DEM data for the AOI;
2. enforce the pre-event cutoff for valid imagery;
3. preprocess and normalize spectral and terrain features;
4. build patch samples with a 64x64 window at 10 m pixel spacing;
5. rasterize and validate landslide polygons in the same CRS;
6. create spatially separated training and test folds with a buffer between clusters;
7. train baseline and comparison models with the same feature inventory;
8. compute segmentation metrics and error maps;
9. build a georeferenced susceptibility output and documented diagnostics.

## Configured values

The checked-in YAML files define the intended configuration:

- CRS: EPSG:32644 (UTM 44N)
- Target resolution: 10 m
- Patch size: 64 x 64
- Patch stride: 64
- Input channels: 16
- Feature groups: 10 Sentinel-2 bands, NDVI, BSI, MNDWI, slope, aspect, curvature
- Loss: focal + dice loss
- Spatial CV: eight geographically clustered folds
- Buffer policy: buffer zones are intended to separate training and test regions

## Scientific limitations and constraints

1. The project requires real Sentinel-2 and DEM data from the Copernicus ecosystem. Without these, the model cannot be trained on valid observations.
2. Cloud and invalid pixels must be masked and excluded from training and evaluation. A silent fallback to non-landslide values would invalidate the results.
3. The DEM resolution is approximately 30 m native resolution. Resampling to 10 m does not create new terrain information; it only changes the representation of the existing terrain surface.
4. The 30 m minimum mappable unit is a real limitation. Small landslides should not be hidden by the primary mask without being tracked separately.
5. The repository currently contains a synthetic demo and not a real-data benchmark; any final publication claim must come after real training and fold-level evaluation.

## Recommended next steps for a journal-grade run

- Freeze the exact dataset version and acquisition date list.
- Log each image tile and its cloud mask statistics.
- Persist the buffer-based cluster assignments for each fold.
- Save the training config, random seed, label masks, and checkpoint versions together.
- Keep a separate set of development smoke tests from the final experiment records.
- Publish the final metrics per fold, not only the best-performing fold.
- Report an ablation study that isolates the effect of terrain and spectral indices.

## Current repository status

The repository is a strong scientific prototype, but it should be treated as a development and design project until a real-data training run has produced reproducible metrics. At present, the evidence is sufficient for research engineering and method design, but not sufficient for a publication-ready performance claim.

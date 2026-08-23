# Architecture & Design Rationale

Aggregates the reasoning behind every non-obvious decision in `notebooks/landslide_pipeline.ipynb`, in one place, instead of leaving it scattered across docstrings. This is a rationale document, not a literature survey — external repos/papers are deliberately not cited here with invented specifics; see the "Out of scope" note at the end.

## Framing: susceptibility, not raw "forecasting"

The ACCIMT inventory is a post-event damage-mapping dataset by its own stated purpose (`docs/data_card.md`). Training a "forecasting" model on it is only valid if every model input is drawn from strictly pre-event imagery/terrain — otherwise the model just re-detects a scar's own already-visible signature. This is why Step 2 mechanically verifies every downloaded Sentinel-2 timestamp against the cutoff (`verify_pre_event_dates`) instead of trusting the request-side date filter alone.

## Why UTM 44N (EPSG:32644), forced explicitly

The source labels are WGS84 geographic (EPSG:4326); all patch sizing, buffer distances, and slope/aspect math need real meters, not degrees. Both `download_sentinel2` and `download_dem` call `resample_spatial` to force this CRS at the source — Copernicus DEM in particular is natively EPSG:4326, so skipping this step would silently corrupt every downstream metric-grid assumption.

## Why an 8-10 band subset, not all 13

Per the assignment guidance doc's cited Sequential Forward Floating Selection research: a compact visible + red-edge + NIR + SWIR subset matches or beats a full 13-band input while avoiding the Hughes phenomenon (curse of dimensionality from highly correlated bands).

## Why 256x256 @ 10m patches, stride 128

Derived from the real polygon size distribution (`docs/data_card.md`), not assumed: 256px @ 10m = 2.56km comfortably contains even the p99 polygon (917m) with room for context. 50% stride overlap keeps edge-adjacent polygons fully inside the *next* tile too.

## Why a 30m minimum mappable unit

16.0% of labeled polygons have a minor-axis under 30m — near or below what Sentinel-2's 10-20m bands can resolve, and independently corroborated by the source shapefile's own processing lineage (ArcGIS `SmoothPolygon`, 20m PAEK tolerance — sub-20m boundary precision was never present in the ground truth to begin with). Polygons below this threshold are excluded from the primary training mask and reported as a separate small-object cohort, not silently dropped or blended in.

## Why `all_touched=True` on rasterization

The alternative default (burn only if a polygon covers a pixel's center) can make a sub-pixel polygon vanish from the mask entirely at an unlucky sub-pixel offset — exactly the population the minimum-mappable-unit split exists to track honestly.

## Why clustered + buffered spatial CV, never random k-fold

No district/date/severity field exists anywhere in the source data (`docs/data_card.md`), so clusters are computed (K-Means on centroids) rather than read from an attribute. Random k-fold on adjacent patches leaks via spatial autocorrelation (Tobler's First Law) and produces inflated, dishonest validation metrics. The buffered margin strips residual cross-boundary leakage that plain block CV would still allow.

## Why negatives are slope-stratified (density-proportional) plus hard negatives, not random

Most terrain is flat; random negatives would be trivially separable by slope alone and would never test the model on genuinely ambiguous terrain. Density-proportional slope-bin stratification matches the *shape* of the positive slope distribution, not just its range. Hard negatives — drawn from immediately outside the exclusion buffer but inside a slightly larger ring — are adjacent to real failures (same rainfall event, same rough geomorphology) without overlapping them, forcing the model to learn real discriminative signal instead of "steep plus wet."

## Why Focal+Dice loss

A median positive patch is ~0.05% landslide pixels (`docs/data_card.md`) — plain BCE/accuracy would let the optimizer converge to "predict nothing everywhere" at 95%+ "accuracy." Focal loss down-weights easy background pixels; Dice directly optimizes overlap, which Focal alone doesn't reward.

## Why two architectures, not more

U-Net (baseline) and DeepLabV3+ (attention/multi-scale comparison), both via `segmentation_models_pytorch`, both on the same input stack. Everything beyond these two — Swin transformers, bitemporal Siamese change-detection, SAR fusion — is deliberately literature-review/discussion scope for this project's timeline, not implementation scope; see the locked scope decisions in the project plan.

## What's real vs. synthetic-verified-only

See the notebook's own Step 6 status table. In short: label loading and spatial clustering/CV are verified against the real 4,225-polygon shapefile; everything downstream of actual pixel data (acquisition, preprocessing, sampling, modeling, evaluation) is correctly implemented and self-checked against synthetic fixtures only, because no real Sentinel-2/DEM imagery has been downloaded yet.

## Out of scope for this document

A GitHub-repository survey, an academic literature review, and a full algorithm comparison table are legitimate parts of the eventual written report, but are not fabricated here — producing them responsibly requires real searches against real sources at the time they're written, not invented citations placed here for completeness.

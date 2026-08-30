# Architecture & Model Design

Design rationale for `notebooks/landslide_pipeline.ipynb`, kept short — one line per decision, not an essay.

## Pipeline decisions

- **CRS: UTM 44N (EPSG:32644), forced.** Source labels are WGS84; patch sizing/buffers/slope math need meters. Both downloads force this CRS at the source — Copernicus DEM is natively EPSG:4326 otherwise.
- **30-day pre-event search window.** Shorter lookback = imagery closer to the event, less vegetation drift. Tradeoff: less redundancy against cloud gaps.
- **8-10 band subset, not all 13.** Visible + red-edge + NIR + SWIR avoids the Hughes phenomenon (SFFS research, assignment guidance doc).
- **256x256 @ 10m patches, stride 128.** Covers the p99 polygon (917m) with context; derived from the real size distribution, not assumed.
- **30m minimum mappable unit.** 31% of polygons are under this by true oriented minor axis (near/below Sentinel-2 resolution, and the source shapefile's own 20m PAEK smoothing tolerance). Excluded from the primary mask, tracked as a separate small-object cohort.
- **`all_touched=True` rasterization.** Default (pixel-center-only) can drop a sub-pixel polygon entirely.
- **Clustered + buffered spatial CV, never random k-fold.** No district/date field exists in the source data, so clusters are K-Means on centroids. Random k-fold leaks via spatial autocorrelation.
- **Slope-stratified + hard negatives, not random.** Random negatives are trivially separable by slope alone. Scope reduction: slope only, not aspect/elevation/land-cover jointly.
- **Terrain: slope/aspect (finite-difference) + curvature (2nd-difference).** Simple, defensible, not a full Horn/Zevenbergen-Thorne kernel. TWI out of scope (needs flow-accumulation routing).
- **MNDWI over NDWI; no NBR.** MNDWI separates water from bare/built-up better (relevant since exposed scars can look like water). NBR is a burn-severity index with no landslide-susceptibility role.
- **Focal+Dice loss.** ~0.05% positive-pixel imbalance means plain BCE/accuracy converges to "predict nothing."
- **`encoder_weights="imagenet"` caveat.** 16-channel input vs ImageNet's 3 — only ~3 channels' first-layer weights are genuinely pretrained, the rest start near-random despite the "pretrained" label.

## Model architecture

**U-Net (baseline) + DeepLabV3+ (comparison)**, both `segmentation_models_pytorch`, ResNet34 encoder, same 16-channel input stack (10 bands + NDVI/BSI/MNDWI + slope/aspect/curvature).

Rejected alternatives:
- **Pixel-wise RF/XGBoost** — no spatial context, can't produce smooth boundaries. Viable only as a sanity-check baseline.
- **Patch-level classification** — throws away boundary precision the MMU/rasterization work was for; incompatible with mIoU.
- **ViT/Swin** — needs a large corpus or a pretrained remote-sensing checkpoint, neither available here.

## Literature grounding (real sources, searched 2026-08-23)

- [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022) — closest benchmark: multispectral+terrain, pixel-wise, ~10m. Confirms this design pattern; doesn't enforce pre-event-only like this project does.
- [DeepLabV3+ vs U-Net comparison](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10674776/) — 80.28% vs 76.2% mIoU on their dataset; consistent with, not proof for, the choice here.
- [Unified Focal loss](https://arxiv.org/abs/2102.04525) — Focal+Dice is a standard response to severe imbalance, not idiosyncratic.
- [Spatial CV for GeoAI](https://www.acsu.buffalo.edu/~yhu42/papers/2023_GeoAIHandbook_SpatialCV.pdf) — random k-fold on spatial data is a well-established leakage risk.
- [SFFS band selection](https://arxiv.org/html/2605.09746) — supports the reduced band subset over full 13-band input.

Not an exhaustive systematic review — a grounding survey for this project's own decisions.

## Planned ablation (not yet run)

Pre-event vs. post-event counterfactual: train the same architecture twice, once obeying the pre-event constraint and once deliberately violating it. Expected result: the post-event arm scores measurably higher (it can see the scar directly) — a result where it doesn't would itself be a finding worth investigating. Needs a second acquisition run with `pre_event_cutoff` inverted, kept in a separate `DATA_DIR` path so it can never leak into the real run.

## What's real vs. synthetic-verified

See the notebook's own Step 6 status table — per-component, changes independently, not repeated here.

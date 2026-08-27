# Architecture & Design Rationale

Aggregates the reasoning behind every non-obvious decision in `notebooks/landslide_pipeline.ipynb`, in one place, instead of leaving it scattered across docstrings. This is a rationale document, not a literature survey — external repos/papers are deliberately not cited here with invented specifics; see the "Out of scope" note at the end.

## Framing: susceptibility, not raw "forecasting"

The ACCIMT inventory is a post-event damage-mapping dataset by its own stated purpose (`docs/data_card.md`). Training a "forecasting" model on it is only valid if every model input is drawn from strictly pre-event imagery/terrain — otherwise the model just re-detects a scar's own already-visible signature. This is why Step 2 mechanically verifies every downloaded Sentinel-2 timestamp against the cutoff (`verify_pre_event_dates`) instead of trusting the request-side date filter alone.

## Why UTM 44N (EPSG:32644), forced explicitly

The source labels are WGS84 geographic (EPSG:4326); all patch sizing, buffer distances, and slope/aspect math need real meters, not degrees. Both `download_sentinel2` and `download_dem` call `resample_spatial` to force this CRS at the source — Copernicus DEM in particular is natively EPSG:4326, so skipping this step would silently corrupt every downstream metric-grid assumption.

## Why a 30-day pre-event search window, not 90

A shorter lookback keeps imagery closer to the actual event, reducing land-cover/vegetation drift versus a longer window — a freshness argument, not just a speed one. Real tradeoff, stated rather than hidden: fewer candidate dates means less redundancy against cloud gaps, so some sub-areas could in principle return zero qualifying scenes where a longer window would have found one. Each acquisition run confirms at least one date per AOI tile before proceeding.

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

**Scope limitation, stated explicitly:** stratification matches slope only, not aspect/elevation/land-cover jointly — a real reduction from the original plan's fuller terrain-matching, not an equivalent design. Slope is the single strongest, cheapest-to-compute discriminator available without a land-cover raster (which this project doesn't acquire), so it was kept as the one dimension actually implemented — adding the others is a documented future extension, not a silent omission.

## Terrain features: slope, aspect, curvature — and why not more

`slope_aspect()` (finite-difference gradient) and `curvature()` (second-difference Laplacian, sign-flipped so ridges are positive and valleys negative) are both simple, closed-form, and match the "an undergraduate can implement and defend this" bar this project is held to. Neither is a full Zevenbergen & Thorne (1987) or Horn (1981) 3x3 kernel — those are more accurate but harder to derive and verify by hand; the coarser approximation is adequate for stratifying negative samples and giving the model terrain context, not for precision geomorphometry. TWI remains out of scope (flow-accumulation routing) — see `docs/limitations.md`.

## Why MNDWI instead of NDWI, and why not NBR

MNDWI (Modified NDWI, using SWIR instead of NIR) was chosen because it separates water from built-up/bare surfaces more cleanly than the original NDWI — relevant here because exposed, saturated landslide scars can otherwise be confused with water by plain NDWI. NBR (Normalized Burn Ratio) was considered and deliberately excluded: it is a fire/burn-severity index with no established role in landslide susceptibility literature, and adding it just to superficially match a generic remote-sensing feature checklist would be feature engineering without a stated mechanism — the kind of thing a viva panel should reject, not reward.

## A real caveat on `encoder_weights="imagenet"` with a 16-channel input

Both architectures load an ImageNet-pretrained `resnet34` encoder but take a 16-channel input, not ImageNet's native 3 (RGB). `segmentation_models_pytorch` handles the shape mismatch (its `in_channels` argument reshapes the first conv layer), but the pretrained filters are only meaningful for whichever 3 channels approximate RGB — the other 13 channels' first-layer weights are effectively randomly initialized despite `encoder_weights="imagenet"` being set. This is a real, partial-transfer-learning limitation, not a bug: the encoder still benefits from ImageNet's deeper, more generic feature hierarchy, but the "pretrained" label overstates how much of the first layer is actually pretrained for this specific input stack. Worth remembering when comparing the baseline's performance against a hypothetical from-scratch encoder — the gap may be smaller than a naive "pretrained vs. not" framing would suggest.

## Why Focal+Dice loss

A median positive patch is ~0.05% landslide pixels (`docs/data_card.md`) — plain BCE/accuracy would let the optimizer converge to "predict nothing everywhere" at 95%+ "accuracy." Focal loss down-weights easy background pixels; Dice directly optimizes overlap, which Focal alone doesn't reward.

## Why two architectures, not more

U-Net (baseline) and DeepLabV3+ (attention/multi-scale comparison), both via `segmentation_models_pytorch`, both on the same input stack. Everything beyond these two — Swin transformers, bitemporal Siamese change-detection, SAR fusion — is deliberately literature-review/discussion scope for this project's timeline, not implementation scope; see the locked scope decisions in the project plan.

## What's real vs. synthetic-verified-only

See the notebook's own Step 6 status table. In short: label loading, the label-geometry EDA (minor-axis/area histograms, cluster-balance check), and spatial clustering/CV are verified against the real 4,225-polygon shapefile; everything downstream of actual pixel data (acquisition, preprocessing, sampling, modeling, evaluation, per-cluster metric aggregation) is correctly implemented and self-checked against synthetic fixtures only, because no real Sentinel-2/DEM imagery has been downloaded yet.

## Model comparison: why U-Net + DeepLabV3+ beat the alternatives for this task

Evaluated against this project's actual constraints (irregular polygon targets, ~4,225 labels most under 100 m minor-axis, CPU-capable dev environment plus Colab free-tier GPU, mIoU/pixel-level evaluation mandated by governance rules):

- **Pixel-wise Random Forest / XGBoost:** fast, interpretable, no GPU needed — a reasonable *baseline-of-a-baseline*, but each pixel is classified independently of its neighbors, so the model cannot use shape/texture/context at all. It would need hand-engineered neighborhood features to compensate, still can't produce a smooth boundary, and the extreme class imbalance (~0.05% positive) requires heavy class-weighting that pixel-wise trees handle worse than a loss designed for it (Focal+Dice). Rejected as the primary model; viable as a quick sanity-check baseline if time allows.
- **Patch-level image classification (ResNet/EfficientNet):** mismatches the task directly — the whole point of the minimum-mappable-unit analysis and `all_touched=True` rasterization was boundary-level precision; collapsing that to one label per patch throws it away, and patch-edge landslides get miscounted or truncated ("edge effect"). Also incompatible with the mIoU metric the governance rules require.
- **Semantic segmentation (U-Net, DeepLabV3+):** natively pixel-precise, handles irregular/elongated polygon shapes, and DeepLabV3+'s atrous spatial pyramid gives multi-scale context that suits a dataset whose polygon width spans 10 m to 3.5 km. **Chosen** — see "Why two architectures" above for why exactly these two.
- **Vision Transformers (ViT, Swin):** typically need either a large labeled corpus or a strong pretrained remote-sensing checkpoint to be worth their compute cost; this project has neither a large corpus (most usable positives are well under 1,000 after the MMU filter) nor a verified, appropriately-licensed pretrained segmentation transformer checkpoint to responsibly cite. Higher compute cost for an unproven benefit at this project's scale and timeline. Rejected for implementation; a legitimate discussion-only item for the write-up's future-work section.

**Verdict, unchanged from the original scope lock:** U-Net (baseline) + DeepLabV3+ (comparison), both via `segmentation_models_pytorch`, both on the same 16-channel input stack.

## Out of scope for this document

A GitHub-repository survey, an academic literature review, and a full algorithm comparison table are legitimate parts of the eventual written report, but are not fabricated here — producing them responsibly requires real searches against real sources at the time they're written, not invented citations placed here for completeness.

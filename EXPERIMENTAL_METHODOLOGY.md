# Experimental Methodology (Corrected, Fold 0)

## Task framing
This is a **pixel-level segmentation / susceptibility** experiment: the model is asked to segment
locations matching the ACCIMT post-event inventory from pre-event Sentinel-2 + terrain inputs. It
is **not** a forecasting experiment — there is no defined forecast horizon, no predictors proven
available at an operational decision time, and no labels for subsequent (post-training-window)
events. Any language elsewhere in the repo implying "forecasting" should be read as susceptibility
mapping instead; this document and the paper derived from it use susceptibility/segmentation
terminology only.

## Data
- ACCIMT Cyclone Ditwah (Nov 2025) inventory, 4,225 polygons, reprojected to EPSG:32644.
- Minimum mapping unit: 30 m oriented minor axis (corrected formula — see AUDIT_REPORT.md Issue 3).
  **3,938 primary polygons** (4,225 − 1,287 below-MMU small objects... note: small-object count and
  primary count are measured on the full inventory; per-cluster/per-tile counts used in training
  are smaller subsets — see EXPERIMENTAL_RESULTS.md for the exact fold-0 figures).
- Sentinel-2 L2A, 10 bands (B02–B12 subset) + SCL, pre-event only (< 2025-11-28), 3 dates available.
- Copernicus DEM GLO-30, resampled to 10 m.
- 16-channel feature stack: 10 spectral + NDVI/BSI/MNDWI + slope/aspect/curvature, unchanged from
  the original design.

## Spatial validation
8-cluster K-Means on polygon centroids (seed 42), 3,060 m buffer excluding any training polygon
within that distance of the held-out cluster. **Fold 0 only was run this session** — see
AUDIT_REPORT.md Issue 8 for why (compute cost, not a methodological choice).

## Masking
Cloud shadow (SCL 3), unclassified (7), cloud medium/high probability (8/9), thin cirrus (10),
no-data (0), saturated/defective (1), and snow/ice (11) pixels are excluded from both the training
loss and every reported metric (corrected — see AUDIT_REPORT.md Issue 4).

## Model / loss / optimizer
U-Net and DeepLabV3+, ResNet34 encoder (ImageNet-initialized, first conv layer adapted to 16
channels), focal-Dice loss (γ=2.0, α=0.25, Dice weight=0.5), now mask-aware. Adam, lr=3e-4,
batch=32. Early stopping (patience 3) on inner-validation loss; best-checkpoint-by-val-loss saved.

## Inner validation
Training-cluster polygons are split 85/15 by `polygon_id` (grouped, not patch-random) before patch
extraction, so no polygon contributes patches to both the fit and inner-validation sets.

## Threshold
Fixed at 0.5 for all reported binary metrics (precision/recall/F1/IoU/specificity). PR-AUC and
ROC-AUC use continuous probabilities and are threshold-independent. Validation-selected threshold
search was not implemented this pass (AUDIT_REPORT.md Issue 11).

## What changed vs. the original (uncorrected) numbers
See AUDIT_REPORT.md and FIXES_APPLIED.md for the full list. The two changes that materially alter
*which patches enter the evaluation* are: (1) the corrected minor-axis formula changes which
polygons count as "primary" (≥30m) vs. small-object, shrinking the fold-0 primary test cohort from
438 to a smaller, corrected set; (2) the cloud mask now actually excludes ~1-2% of pixels from the
metric denominators that were previously counted as valid. Do not directly compare the numbers in
EXPERIMENTAL_RESULTS.md to any numbers from before this pass as if only the checkpoint changed —
the evaluation population itself changed too.

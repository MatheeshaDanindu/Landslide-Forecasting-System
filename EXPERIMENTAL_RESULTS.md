# Experimental Results — Fold 0, Corrected Pipeline

**Status: PARTIAL.** One of eight spatial folds. Do not read as a generalization claim —
see AUDIT_REPORT.md Issue 8 for why only one fold was run (compute cost: ~8,393×7,699 px tiles,
CPU-only this session).

## What produced these numbers
`PYTHONPATH=src python -m landslide_pipeline.evaluate --fold 0 --model-key {baseline,comparison}`,
using the pre-existing repo checkpoints (`Trained Model/fold_cluster0_{baseline,comparison}.pt` —
trained under the *old*, uncorrected pipeline) run through the **corrected evaluation path**:
cloud/shadow/nodata-masked metrics, corrected minor-axis MMU filtering, PR-AUC + ROC-AUC added.
A from-scratch retrain with the corrected masked loss was only smoke-tested (see below), not
completed to convergence — those numbers are not reported as a result.

## Table 1 — Fold 0, pixel-level metrics, threshold = 0.5, cloud/shadow/nodata-masked

| Metric | U-Net (baseline) | DeepLabV3+ (comparison) |
|---|---|---|
| Precision | 0.129 | 0.499 |
| Recall | 0.084 | 0.003 |
| Specificity | 0.961 | 0.9998 |
| F1 | 0.101 | 0.006 |
| IoU | 0.053 | 0.003 |
| PR-AUC (average precision) | 0.094 | 0.112 |
| ROC-AUC | 0.562 | 0.593 |
| TP / FP / FN / TN | 6,877 / 46,564 / 75,340 / 1,139,563 | 243 / 244 / 81,974 / 1,185,883 |
| Test patches | 315 | 315 |
| Valid pixels (post-mask) | 1,268,344 / 1,290,240 (98.3%) | 1,268,344 / 1,290,240 (98.3%) |

Raw CSV: `experiments/corrected_experiment_v1/fold_metrics.csv`.

## Interpretation
- Both models score above the ROC-AUC chance baseline of 0.5 (U-Net +0.062, DeepLabV3+ +0.093) —
  a weak but non-trivial pre-event discriminative signal is present.
- PR-AUC (the metric that matters most under ~6% positive-pixel prevalence) is low for both
  (0.094, 0.112) — far below the ROC-AUC gap alone would suggest, confirming the audit's point
  that ROC-AUC can hide weak positive retrieval under severe imbalance.
- DeepLabV3+ collapses to near-zero positive predictions at threshold 0.5 (243 TP vs. 81,974 FN) —
  consistent with an under-trained model (5 epochs, old uncorrected pipeline) rather than evidence
  against the architecture itself.
- The **test-patch count dropped from 439 (previous, uncorrected run) to 315** after the minor-axis
  fix changed which cluster-0 polygons count as primary (≥30m) vs. small-object. This is expected
  and correct, not a data-loss bug — see EXPERIMENTAL_METHODOLOGY.md.

## Smoke test of the corrected training path (not a result)
`PYTHONPATH=src python -m landslide_pipeline.train --fold 0 --model-key baseline --epochs 2 --max-train-polygons 40`
ran successfully end to end: 34 fit / 6 val / 315 test patches, grouped polygon-level inner split,
masked focal-Dice loss, early-stopping logic exercised, best-checkpoint saved.
Train loss: 0.6425 → 0.6235 over 2 epochs (decreasing, as expected). This demonstrates the
corrected pipeline *runs correctly*; it is **not** a trained model worth evaluating (40 polygons,
2 epochs) and its checkpoint is not included in Table 1.

## What a full corrected run still requires
- Retrain all 8 folds × 2 architectures with the masked loss, to convergence (realistically hours
  on CPU, much less on GPU — not attempted this session).
- Port slope-stratified + hard-negative sampling into `dataset.py`/`train.py` (currently
  positive-centred patches only).
- A fair pixel-wise Random Forest baseline on the same folds/masks.
- Fold-level mean ± SD and a cluster-aware bootstrap CI once more than one fold exists.

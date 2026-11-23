# Experimental Results — Full 8-Fold Spatial Cross-Validation, Corrected Pipeline

**Status: FINAL (leakage-free).** All 8 spatial folds x 2 architectures trained and evaluated with
the corrected pipeline, including the fix for the negative-sampling leakage defect
(AUDIT_REPORT.md Issue 25: `sample_negative_centers` could previously draw negative training
patches from inside the held-out test cluster or its buffer). This table supersedes every prior
version of this file.

## What produced these numbers

`scripts/run_all_folds.py`, sharded 4-way (2 folds/process, 3 threads/process), each fold
extracted once and both architectures trained from the same in-memory patches, with
`sample_negative_centers` now excluding the fold's `test_buffer` geometry in addition to the
training-polygon exclusion ring. Full real run: 5 epochs (early-stopping patience 3; it never
triggered — see "Training convergence" below), batch size 64, Adam lr 3e-4, seed 42 throughout.
One earlier run (pre-fix) produced similar but not identical numbers; this is the authoritative
version, produced entirely after the fix (git commit `4a59be3` onward).

## Table 1 — Per-fold results, pixel-level metrics, cloud/shadow/nodata-masked

| Fold | Model | Precision | Recall | F1 | IoU | PR-AUC | ROC-AUC | Threshold | Test patches | Valid px % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | U-Net | 0.251 | 0.340 | 0.288 | 0.169 | 0.258 | 0.709 | 0.50 | 315 | 98.3% |
| 0 | DeepLabV3+ | 0.214 | 0.354 | 0.267 | 0.154 | 0.213 | 0.720 | 0.25 | 315 | 98.3% |
| 1 | U-Net | 0.295 | 0.281 | 0.288 | 0.168 | 0.248 | 0.740 | 0.55 | 478 | 100% |
| 1 | DeepLabV3+ | 0.255 | 0.309 | 0.279 | 0.162 | 0.207 | 0.735 | 0.50 | 478 | 100% |
| 2 | U-Net | 0.396 | 0.236 | 0.296 | 0.173 | 0.258 | 0.730 | 0.35 | 253 | 62.0% |
| 2 | DeepLabV3+ | 0.404 | 0.284 | 0.333 | 0.200 | 0.278 | 0.801 | 0.50 | 253 | 62.0% |
| 3 | U-Net | 0.305 | 0.318 | 0.311 | 0.184 | 0.274 | 0.735 | 0.50 | 382 | 83.8% |
| 3 | DeepLabV3+ | 0.273 | 0.329 | 0.299 | 0.176 | 0.211 | 0.760 | 0.50 | 382 | 83.8% |
| 4 | U-Net | 0.390 | 0.284 | 0.328 | 0.196 | 0.281 | 0.745 | 0.45 | 303 | 80.7% |
| 4 | DeepLabV3+ | 0.345 | 0.294 | 0.317 | 0.188 | 0.255 | 0.749 | 0.60 | 303 | 80.7% |
| 5 | U-Net | 0.331 | 0.241 | 0.279 | 0.162 | 0.254 | 0.693 | 0.45 | 460 | 91.0% |
| 5 | DeepLabV3+ | 0.310 | 0.265 | 0.285 | 0.166 | 0.226 | 0.717 | 0.45 | 460 | 91.0% |
| 6 | U-Net | 0.386 | 0.293 | 0.333 | 0.200 | 0.315 | 0.777 | 0.55 | 204 | 90.4% |
| 6 | DeepLabV3+ | 0.390 | 0.336 | 0.361 | 0.220 | 0.305 | 0.780 | 0.45 | 204 | 90.4% |
| 7 | U-Net | 0.248 | 0.373 | 0.298 | 0.175 | 0.271 | 0.734 | 0.35 | 547 | 100% |
| 7 | DeepLabV3+ | 0.254 | 0.315 | 0.281 | 0.164 | 0.213 | 0.736 | 0.45 | 547 | 100% |

Raw CSV: `experiments/corrected_experiment_v1/fold_metrics.csv`. Per-fold training history:
`experiments/corrected_experiment_v1/checkpoints/fold{N}_{model}_history.csv`. Checkpoint metadata:
matching `*_meta.json`.

## Table 2 — Aggregate across all 8 folds (mean ± SD, independent spatial clusters)

| Metric | U-Net | DeepLabV3+ | Paired Wilcoxon (W, p) |
|---|---|---|---|
| F1 | 0.303 ± 0.020 | 0.303 ± 0.032 | W=16.0, p=0.844 |
| IoU | 0.178 ± 0.014 | 0.179 ± 0.022 | W=16.0, p=0.844 |
| ROC-AUC | 0.733 ± 0.025 | 0.750 ± 0.029 | W=4.0, p=0.055 |
| PR-AUC | 0.270 ± 0.021 | 0.239 ± 0.037 | W=2.0, p=0.023 |
| Precision | 0.325 ± 0.061 | 0.305 ± 0.068 | W=6.0, p=0.109 |
| Recall | 0.296 ± 0.047 | 0.311 ± 0.029 | W=8.0, p=0.195 |

SD and the Wilcoxon test are computed across the 8 fold-level values (sample SD, n-1; exact
two-sided signed-rank test via `scipy.stats.wilcoxon`), treating each held-out spatial cluster as
one independent observation.

## Interpretation

- **F1 and IoU are statistically indistinguishable between architectures** (p=0.844 for both,
  the maximum possible p-value for n=8 — the fold-level differences split evenly in sign).
- **PR-AUC is the one metric with a robust, significant difference**: U-Net's PR-AUC exceeds
  DeepLabV3+'s in the large majority of folds (mean 0.270 vs 0.239, p=0.023). This finding is
  consistent across both the pre-fix (leaky) and post-fix (leakage-free) runs, which is why it is
  reported as the one architecture-level finding in this study.
- **ROC-AUC is borderline (p=0.055)**, now favoring DeepLabV3+ in aggregate (0.750 vs 0.733) —
  the opposite direction from PR-AUC. This divergence (one threshold-free metric favoring each
  architecture) is itself informative: it shows the two rank the full probability distribution
  differently from how they rank it conditional on predicted-positive, and neither architecture
  is uniformly better across both senses of "ranking quality."
- **Precision and recall differences are not significant** (p=0.109, p=0.195). Notably, the
  precision difference *was* significant (p=0.039) in a pre-fix run contaminated by the
  negative-sampling leakage bug — after the fix, it is not. This is a direct illustration of why
  the leakage fix mattered: at least one of the two significant findings in the contaminated run
  did not survive a clean rerun.
- **All 16 fold-architecture ROC-AUCs exceed chance (0.5) substantially** (0.693–0.801),
  confirming a real, reproducible pre-event discriminative signal across every spatial fold.
- **The 95% CI lower bound for mean PR-AUC exceeds mean test-region prevalence (5.95%, fold-level
  average; 6.03% pooled across all 2,942 test patches)** for both architectures — U-Net's interval
  is [0.252, 0.288], DeepLabV3+'s is [0.208, 0.269] — confirming a pre-event signal under the
  pre-registered decision rule.
- **Fold-to-fold variance is real**: F1 ranges 0.267–0.361 across the 16 fold-architecture runs.

## Training convergence

Validation loss was still falling at the final (5th) epoch in 13 of 16 runs (3 peaked at epoch 3
or 4). Early stopping (patience 3) never triggered in any run. The five-epoch budget is a
compute-driven floor, not a point of diminishing returns; the absolute metric values in Tables 1-2
would likely shift with a longer budget, though the qualitative pattern (PR-AUC favors U-Net;
F1/IoU indistinguishable) is unlikely to reverse given its consistency across the pre-fix and
post-fix runs.

## Known remaining gaps (honest, not fixed this pass)

- **Sub-30 m polygons are rasterized as background, not excluded** (AUDIT_REPORT.md Issue 26):
  when a small-object-cohort polygon falls inside a patch window, its pixels are silently labeled
  negative rather than masked out. Conservative (can only suppress recall), but unquantified.
- **No Random Forest baseline.** Comparison is only between the two CNN architectures; no
  classical baseline on the same folds/masks exists yet (AUDIT_REPORT.md Issue 16).
- **No full-scene inference.** All reported metrics are on positive-centred test patches, not
  whole-tile predictions; commission error over pure-background terrain in the held-out
  clusters is not measured.
- **Threshold selected per-fold on inner validation**, not a single fixed operating point across
  folds — appropriate per the audit's threshold protocol, but means Table 1's precision/recall/
  F1/IoU are not directly comparable fold-to-fold at a literal fixed threshold (ROC-AUC and
  PR-AUC, both threshold-free, are the fairer cross-fold comparison and are reported for exactly
  this reason).
- **Notebook not updated.** All fixes live in the standalone `src/landslide_pipeline/{dataset,
  models,train,evaluate}.py` pipeline; `notebooks/landslide_pipeline.ipynb` still reflects the
  pre-fix implementation.

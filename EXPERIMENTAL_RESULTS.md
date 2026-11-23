# Experimental Results — Full 8-Fold Spatial Cross-Validation, Corrected Pipeline

**Status: SUPERSEDED — retraining in progress.** The table below was produced before a real
negative-sampling leakage defect (AUDIT_REPORT.md Issue 25) was found and fixed: negative training
patches could be drawn from inside the held-out test cluster or its buffer, because
`sample_negative_centers` only ever excluded a ring around *training* polygons, never the test
region itself. All 16 fold/model combinations below predate that fix and are being retrained with
it; **do not cite these numbers as final.** This file will be updated once the retrain completes.
The methodology description, aggregate-statistics method, and interpretation notes below remain
accurate for the corrected rerun; only the numeric values are expected to shift.

## What produced these numbers

`scripts/run_all_folds.py`, sharded 4-way (2 folds/process, 3 threads/process), each fold
extracted once and both architectures trained from the same in-memory patches. One crash
(fold 0 DeepLabV3+, a BatchNorm batch-size-1 edge case) was fixed (`drop_last=True`) and that
one combination was rerun standalone; every other result is from the original parallel run.
Full real run, not a smoke test: 5 epochs (early-stopping patience 3, none triggered early
stopping — all 16 runs used the full 5-epoch budget), batch size 64, Adam lr 3e-4,
seed 42 throughout.

## Table 1 — Per-fold results, pixel-level metrics, cloud/shadow/nodata-masked

| Fold | Model | Precision | Recall | F1 | IoU | PR-AUC | ROC-AUC | Threshold | Test patches | Valid px % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | U-Net | 0.292 | 0.302 | 0.297 | 0.174 | 0.263 | 0.680 | 0.40 | 315 | 98.3% |
| 0 | DeepLabV3+ | 0.304 | 0.272 | 0.288 | 0.168 | 0.232 | 0.714 | 0.45 | 315 | 98.3% |
| 1 | U-Net | 0.301 | 0.289 | 0.295 | 0.173 | 0.263 | 0.730 | 0.45 | 478 | 100% |
| 1 | DeepLabV3+ | 0.300 | 0.300 | 0.300 | 0.176 | 0.238 | 0.771 | 0.55 | 478 | 100% |
| 2 | U-Net | 0.379 | 0.314 | 0.343 | 0.207 | 0.309 | 0.786 | 0.55 | 253 | 62.0% |
| 2 | DeepLabV3+ | 0.323 | 0.376 | 0.348 | 0.210 | 0.257 | 0.813 | 0.45 | 253 | 62.0% |
| 3 | U-Net | 0.304 | 0.321 | 0.312 | 0.185 | 0.280 | 0.758 | 0.45 | 382 | 83.8% |
| 3 | DeepLabV3+ | 0.266 | 0.341 | 0.299 | 0.176 | 0.208 | 0.764 | 0.50 | 382 | 83.8% |
| 4 | U-Net | 0.395 | 0.252 | 0.308 | 0.182 | 0.281 | 0.760 | 0.60 | 303 | 80.7% |
| 4 | DeepLabV3+ | 0.299 | 0.322 | 0.310 | 0.183 | 0.241 | 0.743 | 0.45 | 303 | 80.7% |
| 5 | U-Net | 0.328 | 0.249 | 0.283 | 0.165 | 0.249 | 0.709 | 0.60 | 460 | 91.0% |
| 5 | DeepLabV3+ | 0.325 | 0.248 | 0.282 | 0.164 | 0.225 | 0.707 | 0.50 | 460 | 91.0% |
| 6 | U-Net | 0.451 | 0.296 | 0.357 | 0.217 | 0.338 | 0.787 | 0.65 | 204 | 90.4% |
| 6 | DeepLabV3+ | 0.397 | 0.296 | 0.339 | 0.204 | 0.276 | 0.735 | 0.50 | 204 | 90.4% |
| 7 | U-Net | 0.272 | 0.321 | 0.294 | 0.173 | 0.261 | 0.740 | 0.50 | 547 | 100% |
| 7 | DeepLabV3+ | 0.210 | 0.307 | 0.250 | 0.143 | 0.168 | 0.714 | 0.45 | 547 | 100% |

Raw CSV: `experiments/corrected_experiment_v1/fold_metrics.csv`. Per-fold training history
(epoch-wise train/val loss): `experiments/corrected_experiment_v1/checkpoints/fold{N}_{model}_history.csv`.
Checkpoint metadata (architecture, patch counts, selected threshold): matching `*_meta.json`.

## Table 2 — Aggregate across all 8 folds (mean ± SD, independent spatial clusters)

| Metric | U-Net | DeepLabV3+ |
|---|---|---|
| F1 | 0.311 ± 0.026 | 0.302 ± 0.032 |
| IoU | 0.185 ± 0.018 | 0.178 ± 0.021 |
| ROC-AUC | 0.744 ± 0.037 | 0.745 ± 0.034 |
| PR-AUC | 0.281 ± 0.028 | 0.231 ± 0.034 |

SD is computed across the 8 fold-level values (sample SD, n-1), treating each held-out spatial
cluster as one independent observation -- the right resampling unit per the audit's
pseudo-replication concern (Issue 10), not a pixel-level bootstrap.

## Interpretation

- **F1, IoU, recall, and ROC-AUC do not differ significantly between architectures**, but
  **PR-AUC and precision do** — this nuance was missed in an earlier draft of this document,
  which overstated the two models as "indistinguishable on every aggregate metric." A paired
  Wilcoxon signed-rank test across the 8 folds (verified with `scipy.stats.wilcoxon`) finds:
  U-Net's PR-AUC is higher in **8 of 8 folds** (mean difference +0.050, exact two-sided
  p = 0.0078), and U-Net's precision is higher in **7 of 8 folds** (p = 0.039). ROC-AUC differs
  by only 0.001 in aggregate (0.744 vs 0.745) and the paired per-fold differences split 4/4 in
  sign, consistent with no difference; F1 differences split 5/3 in sign (p = 0.25, not
  significant). RQ2 is therefore answered with a qualification: the architectures are
  indistinguishable on overall segmentation quality (F1/IoU/ROC-AUC), but U-Net ranks the
  positive class more reliably and produces fewer false positives per true positive (PR-AUC,
  precision) at this training budget.
- **All 8 ROC-AUCs exceed chance (0.5) substantially** (0.680-0.813), confirming a real,
  reproducible pre-event discriminative signal across every spatial fold, not a fluke of one
  cluster.
- **PR-AUC (0.23-0.34) remains well below ROC-AUC** for every fold, the expected signature of
  class imbalance (~6% positive-pixel prevalence): the model ranks positives above negatives
  reasonably well in aggregate, but precise pixel-level retrieval of the rare positive class is
  harder, which F1/IoU (0.25-0.36 / 0.14-0.22) reflect directly.
- **Fold-to-fold variance is real and non-trivial** (F1 ranges 0.250-0.357 across folds for
  either model) -- a single-fold result (as originally reported for fold 0 only) would have
  been a poor estimate of the model's true cross-cluster performance; this is exactly the
  failure mode spatial cross-validation with multiple folds is meant to catch.
- **Valid-pixel fraction varies substantially by fold** (62.0% to 100%), reflecting real
  differences in cloud/shadow/snow contamination across the AOI's four Sentinel-2 tiles --
  not a bug, but worth noting when comparing folds: fold 2's smaller valid-pixel fraction
  (62.0%) means its F1 is computed over fewer, specifically non-cloud-contaminated pixels.

- **Training had not converged.** Validation loss was still falling at the final (5th) epoch in
  14 of 16 runs (verified from the per-run `*_history.csv` files; the other 2 peaked at epoch 4).
  Early stopping (patience 3) never triggered in any run. The five-epoch budget is a compute-driven
  floor, not a point of diminishing returns — a longer budget would likely change the absolute
  numbers in Table 1/2, though not necessarily the F1/IoU/ROC-AUC architecture-indistinguishability
  finding.

## Known remaining gaps (honest, not fixed this pass)

- **Sub-30 m polygons are rasterized as background, not excluded** (AUDIT_REPORT.md Issue 26):
  when a small-object-cohort polygon falls inside a patch window, its pixels are silently labeled
  negative rather than masked out. Conservative (can only suppress recall), but unquantified.
- **No Random Forest baseline.** Comparison is only between the two CNN architectures; no
  classical baseline on the same folds/masks exists yet (AUDIT_REPORT.md Issue 16).
- **No full-scene inference.** All reported metrics are on positive-centred test patches, not
  whole-tile predictions; commission error over pure-background terrain in the held-out
  clusters is not measured (AUDIT_REPORT.md Issue 7 / discussion in the original single-fold
  writeup).
- **Threshold selected per-fold on inner validation**, not a single fixed operating point
  across folds -- appropriate per the audit's threshold protocol, but means Table 1's precision/
  recall/F1/IoU are not directly comparable fold-to-fold at a literal fixed threshold (ROC-AUC
  and PR-AUC, both threshold-free, are the fairer cross-fold comparison and are reported for
  exactly this reason).
- **Notebook not updated.** All fixes live in the standalone `src/landslide_pipeline/{dataset,
  models,train,evaluate}.py` pipeline; `notebooks/landslide_pipeline.ipynb` still reflects the
  pre-fix implementation.

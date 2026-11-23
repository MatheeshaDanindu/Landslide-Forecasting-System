# Fixes Applied

Each entry: issue, file(s) changed, what changed, measured effect where one exists.

## Issue 3 — Minor-axis calculation (CRITICAL)
**File:** `src/landslide_pipeline/labels.py`
`minimum_rotated_rectangle.length / 2.0` is the rectangle's half-perimeter (long + short side), not
the short side. Replaced with the minimum of the rectangle's two distinct edge lengths.
**Measured effect:** small-object cohort (<30 m minor axis) went from 85 polygons (2.0%) to
1,287 polygons (30.5%) on a full re-run of `scripts/check_pipeline.py`. This matches
`docs/architecture.md`'s independently-stated "31%" figure (computed via the notebook's
already-correct version of this calculation), which is strong corroborating evidence the bug was
real and the fix is right.

## Issue 21 — Config merge overwrites keys (HIGH)
**File:** `src/landslide_pipeline/config.py`, `scripts/check_pipeline.py`
`load_project_config()` did `merged.update(data)` across all four YAML files in sequence.
`acquisition.yaml` and `preprocessing.yaml` both define a top-level `labels:` key; the second
`update()` silently discarded `labels.path` from the first. Fixed by returning a namespaced dict
(`cfg["acquisition"]`, `cfg["preprocessing"]`, `cfg["model"]`, `cfg["spatial_cv"]`) so same-named
keys across files can never collide. Updated `check_pipeline.py` to the new access pattern.
**Note:** this bug was silent in practice because `check_pipeline.py`'s hardcoded fallback string
for the label path happened to equal the correct value — it would have surfaced immediately on any
code path that didn't have a matching fallback.

## Issue 22 — Dataset path inconsistency (HIGH)
**Files:** `scripts/check_pipeline.py`, `src/landslide_pipeline/{train,evaluate}.py`
Fixing Issue 21 exposed a second, real inconsistency: `acquisition.yaml`'s `labels.path` is
DATA_DIR-relative (`DATA_DIR = REPO_ROOT/"data"`, per the notebook's own Step 0 setup and
`docs/architecture.md`), but the Sentinel-2/DEM raw tiles are placed directly under
`REPO_ROOT/"raw"`, not `REPO_ROOT/"data"/"raw"`. All three scripts now resolve the label path
under `REPO_ROOT/"data"` explicitly, with a comment explaining why, instead of silently relying on
both paths resolving to the same place by coincidence.

## Issue 5 — SCL exclusion list incomplete (HIGH)
**File:** `configs/preprocessing.yaml`
`scl_exclude` changed from `[3,7,8,9,10]` to `[0,1,3,7,8,9,10,11]`, adding no-data (0),
saturated/defective (1), and snow/ice (11) — snow/ice pixels can read spectrally similar to
exposed bare-soil scars and were a genuine confound risk.

## Issue 4 — Cloud mask discarded before loss (CRITICAL)
**Files:** `src/landslide_pipeline/{dataset,training,metrics,train,evaluate}.py`
Confirmed in the notebook (`x, _ = extract_patch(...)` in `LandslideDataset.__getitem__`) via
direct grep; **not edited in the notebook** (100-cell surgery was out of scope this pass).
Fixed in a new parallel standalone pipeline instead: `dataset.py::build_patch` returns
`(features, label, valid_mask)`; `training.py`'s loss functions and `metrics.py`'s
`segmentation_metrics` both accept and apply the mask (`sum(mask*loss)/sum(mask)` for the loss;
masked pixels dropped before any metric is computed). `train.py`/`evaluate.py` wire this through
end to end.

## Issue 9 — Random patch-level train/val split (CRITICAL)
**File:** `src/landslide_pipeline/train.py`
Confirmed in notebook `train_real()` (`torch.utils.data.random_split`). New `train.py` instead
groups the inner validation split by `polygon_id` before any patch is extracted, so no polygon can
have patches on both sides of the inner split. The outer spatial-CV buffer (3,060 m,
cluster-based) is unchanged and still the only thing separating train from the held-out test
cluster.

## Issue 12 — PR-AUC absent (HIGH)
**File:** `src/landslide_pipeline/metrics.py`
Added `average_precision()` and `roc_auc()`, both pixel-level, valid-mask-aware, NaN-safe for
single-class folds. Both are now reported alongside precision/recall/F1/IoU/specificity in every
call to `segmentation_metrics()`.

## Issue 14 — Focal-Dice loss not mask-aware (HIGH)
**File:** `src/landslide_pipeline/training.py`
`focal_loss`, `dice_loss`, `focal_dice_loss` all accept an optional `valid_mask`. Masked pixels
contribute zero to both the focal cross-entropy term and the Dice overlap term, instead of being
silently treated as real negative observations.

## Issue 15 — Minimal training setup (HIGH, partial)
**File:** `src/landslide_pipeline/train.py`
Added: early stopping (`--patience`, default 3), best-validation-loss checkpoint saving (not just
last epoch), per-epoch CSV history log, checkpoint metadata JSON (architecture, encoder, channel
count, fold, seed, patch counts, best val loss). **Not added:** LR scheduler, data augmentation —
scope cut given the one-fold compute budget available this session; see "Deferred" below.

## Issue 24 — Incomplete seeding (MEDIUM)
**File:** `src/landslide_pipeline/training.py`
`seed_everything()` now also seeds Python's `random` module (previously only numpy/torch/cuda).
Added `worker_init_fn` for `DataLoader(..., num_workers>0)` so forked workers don't replay an
identical RNG stream. CUDA determinism flags (`torch.backends.cudnn.deterministic`) not added —
no GPU was available to verify this pass.

## Issue 8 — Full 8-fold retrain (CRITICAL)
**File:** `scripts/run_all_folds.py`
Previously deferred (compute cost). All 8 folds × 2 architectures were trained and evaluated via
sharded parallel processes extracting each fold's patches once and training both models from them.
Full per-fold table, aggregate mean±SD, and a paired Wilcoxon comparison are in
`EXPERIMENTAL_RESULTS.md`. Superseded by Issue 25 below — these fold/model checkpoints are being
retrained again under the leakage fix.

## Issue 7 (continued) — Negative sampling ported, then found to leak (CRITICAL)
**Files:** `src/landslide_pipeline/{dataset,train}.py`
Slope-stratified negative sampling (`sample_negative_centers`) was ported into the standalone
pipeline and mixed into fit/val. While re-reading this code to answer an external report,
a real leakage defect was found: candidates were drawn from the whole Sentinel-2 tile, excluded
only near training polygons — never excluded from the held-out test cluster or its 3,060 m buffer.
Fixed by passing `test_buffer` into `sample_negative_centers` as an `extra_exclusion_geom`
(see Issue 25 in AUDIT_REPORT.md for the full writeup). **All fold/model combinations trained
before this fix are invalid and are being retrained.**

## Issue 11 — Threshold selection on inner validation (HIGH)
**File:** `src/landslide_pipeline/train.py`
After training, the best checkpoint is reloaded and a 0.05–0.95 threshold sweep runs against the
inner-validation split; the F1-maximizing threshold is frozen and saved to the checkpoint's
`meta.json`. `evaluate.py` reads this value by default. Reported precision/recall/F1/IoU in
`EXPERIMENTAL_RESULTS.md` use this per-fold tuned threshold, not a fixed 0.5 — stated explicitly
in every table so the numbers aren't misread as directly threshold-comparable across folds.

## Issue 13 — Cross-fold statistics (HIGH)
**File:** `EXPERIMENTAL_RESULTS.md`
With all 8 folds trained, mean ± SD (n=8) and a paired Wilcoxon signed-rank test across folds are
now reported and were verified with `scipy.stats.wilcoxon`/`scipy.stats.t`, not hand arithmetic.
Finding: U-Net's PR-AUC is higher in 8/8 folds (p=0.0078) and precision in 7/8 (p=0.039); F1, IoU,
recall, and ROC-AUC do not differ significantly. An earlier draft of EXPERIMENTAL_RESULTS.md
overstated this as "indistinguishable on every aggregate metric" — corrected.

## Issue 1 — No standalone pipeline (MEDIUM)
**New files:** `src/landslide_pipeline/{dataset,models,train,evaluate}.py`
Real, runnable, config-driven CLI entry points:
```
PYTHONPATH=src python -m landslide_pipeline.train --fold 0 --model-key baseline
PYTHONPATH=src python -m landslide_pipeline.evaluate --fold 0 --model-key baseline
```
The notebook was **not** converted to import these — that's a separate, larger refactor. The two
pipelines currently diverge (see AUDIT_REPORT.md Issue 2); this one is the corrected reference
implementation going forward.

---

## Deferred (confirmed or plausible, not fixed this pass — with reason)

- **Hard-negative ring (40–150 m)** from the notebook was not ported — only the slope-stratified
  exclusion-ring negatives exist in the standalone pipeline.
- **Random Forest fair baseline** (Issue 16) — not built. Needs a parallel tabular-feature
  extraction path at the same per-pixel grain, same folds, same masks.
- **Sub-30 m polygons rasterized as background** (Issue 26, new) — never excluded from the
  loss/metrics when they fall inside a patch window; only ever conservative (can suppress recall,
  not inflate it), but undocumented in the per-fold tables.
- **Per-sample acquisition manifest** (Issue 18) — not built as a persisted file. Spot-checked
  instead: all Sentinel-2 filenames used (`openEO_2025-10-31Z`, `openEO_2025-11-02Z`,
  `openEO_2025-11-20Z`) are confirmed < the 2025-11-28 cutoff by direct string comparison.
- **DEM nodata handling** (Issue 20) — the specific "zero-initialized destination array" bug
  described lives in the notebook's `load_dem_aligned()`, which the standalone `dataset.py` doesn't
  use (it reads each DEM tile window directly). DEM nodata pixels are still not explicitly masked
  in the new path either — flagged, not fixed.
- **Notebook itself** — not edited. Every fix above lives in the new standalone package; the
  notebook remains the old, unfixed code path until someone ports these changes into it or retires
  it in favor of the CLI scripts.
- **Aspect sin/cos encoding, normalization-fit-on-train-fold-only** (Issue 6) — not applicable as
  stated (no z-score normalization exists to leak in the first place) and not added (would change
  `input_channels` 16→17, invalidating every existing checkpoint's first conv layer — a deliberate
  larger change, not a quick fix).

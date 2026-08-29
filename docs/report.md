# Project Report — Vision-Based Landslide Susceptibility Mapping

Consolidated write-up for the ACCIMT Cyclone Ditwah landslide susceptibility project. This document ties together the rationale, methodology, self-review, and current status that otherwise live scattered across `docs/*.md` and the notebook — written as one piece for submission/review, not a new source of truth (each section links to the file that *is* the source of truth for that claim).

## 1. Framing and scope

This is a **susceptibility mapping** system, not raw "forecasting" and not post-event detection. The ACCIMT Cyclone Ditwah inventory (4,225 polygons, `docs/data_card.md`) is a post-event damage-assessment dataset by its own stated purpose — using it responsibly for a predictive task means every model *input* (Sentinel-2 imagery, DEM-derived terrain) is drawn strictly from before the event (2025-11-28), with the labels used only as targets. ACCIMT's own metadata explicitly endorses this framing: "the best predictor of future landslides is the location of past landslides... past landslide maps serve as reliable reference data for training and validating landslide prediction models" (`docs/data_card.md`).

Full rationale for every non-obvious pipeline decision (CRS choice, band subset, patch size, minimum mappable unit, spatial CV design, negative sampling, loss function, architecture choice) is aggregated in `docs/architecture.md` — not repeated here.

## 2. Methodology summary

Six-step pipeline, implemented end-to-end in `notebooks/landslide_pipeline.ipynb`:

1. **Foundations** — config-driven (`configs/*.yaml`), real ACCIMT shapefile loaded and validated, real exploratory data analysis (polygon size/area distributions, cluster balance).
2. **Data acquisition & preprocessing** — Sentinel-2 (pre-event, CRS-forced) + Copernicus DEM via CDSE openEO; SCL-based cloud/shadow masking; NDVI/BSI/MNDWI spectral indices; slope/aspect/curvature terrain; label rasterization respecting a documented minimum mappable unit; patch tiling and extraction.
3. **Sample generation & spatial CV** — K-Means spatial clustering + buffered cluster folds (never random k-fold); slope-stratified density-proportional negative sampling plus adjacency-ring hard negatives; a persisted provenance manifest making pre-event compliance independently auditable.
4. **Modeling** — U-Net (baseline) and DeepLabV3+ (comparison), both via `segmentation_models_pytorch` on the same 16-channel input stack; Focal+Dice hybrid loss matched to a measured ~0.05% positive-pixel imbalance.
5. **Evaluation** — Precision/Recall/F1/mIoU/ROC-AUC plus the pooled confusion matrix (never accuracy), a categorical error map, and genuine held-out spatial cross-validation (`run_spatial_cv`, 5.5): one model trained per cluster, evaluated only on its own held-out cluster, so reported metrics are never train-and-predict-on-the-same-data.
6. **Write-up** — this document, `docs/literature_survey.md`, `docs/ablation_study.md`, `docs/ai_usage.md`, and the notebook's own real-vs-synthetic status table.

See `docs/literature_survey.md` for how this design compares against real, currently-published prior work (Landslide4Sense, comparative U-Net/DeepLabV3+ studies, spatial-CV leakage literature).

## 3. Critical self-review

The project's own review framework asks a small number of hard questions that any landslide-ML pipeline must be able to answer under scrutiny. Answered here directly, each grounded in a specific pipeline mechanism rather than an assertion:

- **Forecasting vs. detection — which is this?** Susceptibility mapping from strictly pre-event input, mechanically enforced (`verify_pre_event_dates`, Step 2) rather than merely requested. See §1 and `docs/limitations.md`.
- **Is post-event leakage possible?** Checked twice: once at the query stage (`temporal_extent` filter) and once independently after download against the actual file's timestamps — the second check exists specifically because the first alone is an unverified assumption about the API. The counterfactual ablation (`docs/ablation_study.md`) is designed to measure the consequence of this constraint directly, not just assert its importance.
- **Are negative samples geologically defensible?** Slope-stratified (density-proportional, matching the positive distribution's shape) plus hard negatives adjacent to real failures — not random background, which would be trivially separable by slope alone. Documented, real reduction in scope: matches slope only, not the fuller aspect/elevation/land-cover matching originally planned (`docs/architecture.md`).
- **Is spatial/temporal leakage controlled?** Spatial: K-Means clusters held out whole, with a buffered margin stripping cross-boundary autocorrelation (`buffered_cluster_folds`), leakage-checked by assertion against the real shapefile. Temporal: every positive comes from one storm — no CV design can manufacture cross-storm generalization evidence that doesn't exist in the data, and `docs/limitations.md` says so explicitly rather than letting a high validation score imply otherwise.
- **Does this generalize beyond this dataset?** No, and this is stated as a limitation, not hidden: single-storm, single-region data means only generalization to unseen *locations within this extent* is measurable.

## 4. Honest status: what's real vs. what's implemented-but-unverified

See the notebook's own Step 6 table for the authoritative, per-component breakdown — acquisition, dataset assembly, and training are tracked independently there since they don't all reach "real" status at once. Summary: label loading, spatial clustering/CV, and the label-geometry EDA are verified against the real 4,225-polygon shapefile; Sentinel-2/DEM acquisition (full-AOI, tiled), real dataset assembly, and a real (non-smoke-test) training run have all been executed against live downloaded imagery, with the trained model persisted to `models/baseline_unet.pt`. That first real run used the pipeline's earlier 1-epoch placeholder config and, as expected at that setting, mostly failed to detect real positives — `epochs` is now `20` (`configs/model.yaml`) and the genuine held-out evaluation (`run_spatial_cv`, 5.5) is implemented and self-checked; a real run of both at the corrected config is the next concrete step, not yet executed against live data.

## 5. Known gaps, stated rather than hidden

- A real training run at the corrected config (`epochs: 20`, not the 1-epoch placeholder the first real run used) and a real run of `run_spatial_cv`'s held-out evaluation against live data — both implemented and self-checked, neither yet executed against the full real dataset.
- The pre-event-vs-post-event counterfactual ablation (`docs/ablation_study.md`) — designed, not yet run.
- Real error analysis (confusable terrain: bare soil, roads, shadow) — needs real predictions to inspect.
- `docs/literature_survey.md` is a grounding survey, not an exhaustive systematic review.
- The provenance manifest (`data/processed/manifest.csv`) does not yet cover every field the assignment's Stage 3 requires (sample id, coordinates, class label, image dimensions, spatial resolution, cloud info, preprocessing ops) — only product id, sensing date, cluster, and positive/negative flag today.

## 6. Credit and disclaimer

Per ACCIMT's data terms (verbatim in `docs/data_card.md`): research/academic/planning use only, no guarantee of accuracy, independent verification required before operational or legal use, and ACCIMT must be credited in any derived output. This project is a course assignment, not an operational hazard-warning system — see `docs/limitations.md` for the full scope of what this model does not and cannot do.

Satellite data: contains modified Copernicus Sentinel-2 and Copernicus DEM data, accessed via the Copernicus Data Space Ecosystem (CDSE) `openeo` API — service credit, not a claim of endorsement by ESA/CDSE.

# Literature & Prior-Work Survey

Real sources, found via live web search on 2026-08-23 — not recalled from training data and not invented. Numbers below are as reported by their source paper on that paper's own dataset; none are claims about this project's dataset unless stated.

## Benchmark datasets directly comparable to this project's input stack

- **Landslide4Sense** ([GitHub](https://github.com/iarai/Landslide4Sense-2022), [competition paper](https://arxiv.org/pdf/2209.02556)) — the closest existing benchmark to this project's design: 128x128 patches, pixel-wise labels, 14 bands per patch (Sentinel-2 B1-B12 + ALOS PALSAR slope + DEM), all resampled to ~10m. Confirms the core design choice here (multispectral + terrain stack, pixel-wise segmentation) is an established pattern, not a one-off. Difference from this project: Landslide4Sense does not enforce or document a pre-event-only constraint — it's framed as detection/inventory mapping from whatever imagery is available, not susceptibility from strictly pre-event input.
- **HR-GLDD** ([GitHub](https://github.com/kushanavbhuyan/HR-GLDD-A-Global-Landslide-Mapping-Data-Repository)) — a global high-resolution landslide mapping data repository with training code, useful as a reference for repository structure and label conventions.
- **PKLandSeg** (Gilgit-Baltistan, Pakistan) — 3,330 samples, 3m PlanetScope RGB + elevation + NDVI + slope + binary masks at 512x512. Confirms elevation/NDVI/slope as a standard minimal terrain-plus-spectral feature set, matching (at coarser resolution) the band+index+terrain stack built in Step 2 here.

## Architecture choice: U-Net vs. DeepLabV3+

["Comparative Evaluation of State-of-the-Art Semantic Segmentation Networks for Long-Term Landslide Map Production"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10674776/) reports DeepLabV3+ at 80.28% mIoU / 88.29% F1 vs. U-Net at 76.2% mIoU / 85.29% F1 **on their own dataset** — DeepLabV3+'s atrous convolutions giving it a larger effective receptive field than standard U-Net. This is consistent with (not proof for) this project's own architecture rationale in `docs/architecture.md` — DeepLabV3+'s multi-scale context suits a polygon-width range spanning 10m to 3.5km. A separate study, ["A comparative study of loss functions and attention mechanisms in landslide semantic segmentation using U-Net"](https://www.nature.com/articles/s41598-025-31789-2), found attention-augmented U-Net variants can close or reverse that gap — reinforcing that the U-Net-vs-DeepLabV3+ comparison run in this project's own Step 4/5 (once real data exists) is worth doing empirically rather than assuming either architecture wins by default.

## Loss function: Focal + Dice under extreme imbalance

["Unified Focal loss: Generalising Dice and cross entropy-based losses..."](https://arxiv.org/abs/2102.04525) and multiple remote-sensing segmentation surveys (e.g. ["Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook"](https://arxiv.org/html/2312.05391v1)) confirm Focal+Dice hybrids are a standard, not idiosyncratic, response to severe foreground/background imbalance — directly supporting this project's `FocalDiceLoss` (Step 4), chosen for a measured ~0.05% positive-pixel rate (`docs/data_card.md`), not a guessed one.

## Spatial cross-validation and leakage

Confirmed via multiple independent sources (["Spatial cross-validation for GeoAI"](https://www.acsu.buffalo.edu/~yhu42/papers/2023_GeoAIHandbook_SpatialCV.pdf), ["Estimating the Prediction Performance of Spatial Models via Spatial k-Fold Cross Validation"](https://arxiv.org/pdf/2005.14263), [Spatial+ cross-validation](https://www.sciencedirect.com/science/article/pii/S1569843223001887)): random k-fold CV on spatially autocorrelated data produces systematically over-optimistic validation scores, because Tobler's First Law means nearby samples share attributes the model can "peek" at across a random split. This is the exact failure mode `buffered_cluster_folds` (Step 3) is built to prevent — the literature treats this as a well-established, not niche, risk in geospatial ML, reinforcing that it was correct to treat random k-fold as categorically rejected rather than one configurable option among several.

## Feature/band selection

["Sequential Feature Selection for Efficient Landslide Segmentation from Multi-Spectral Data"](https://arxiv.org/html/2605.09746) — directly supports this project's own cited rationale (`docs/architecture.md`) for using a reduced band subset (visible + red-edge + NIR + SWIR) rather than the full 13-band Sentinel-2 stack, to avoid the Hughes phenomenon from highly correlated bands.

## Where this project differs from the surveyed literature

Most surveyed work (Landslide4Sense included) frames the task as post-event **detection/inventory mapping** — using whatever imagery is available, often after the event, to map where landslides already occurred. This project's mechanical `verify_pre_event_dates` check (Step 2) and the pre-event-vs-post-event framing throughout `docs/limitations.md` is a stricter constraint than most of the surveyed baselines actually enforce — worth stating explicitly in any write-up rather than implying this project's setup is directly benchmark-equivalent to Landslide4Sense-style results.

## Scope note

This is a survey of real, currently-available sources — not an exhaustive systematic review. It exists to ground this project's already-made design decisions against prior work, and to identify where this project's constraints (pre-event-only, single-storm dataset, course-project compute budget) diverge from typical benchmark setups. A full systematic literature review with formal inclusion criteria is out of scope here, same as noted in `docs/architecture.md`.

# Scientific & Ethical Limitations

This document is a required part of the project, not an afterthought — see `docs/data_card.md` for the dataset facts these limitations are grounded in.

## Susceptibility model, not a real-time early-warning system

This system predicts static landslide **susceptibility** from pre-event optical imagery and terrain (slope, aspect, TWI). It does not model subsurface hydrology, pore-water pressure, soil stratigraphy, or geological faults — factors that actually govern *when* a slope fails and that satellite sensors cannot see. It also does not ingest real-time rainfall, unlike Sri Lanka's National Building Research Organisation (NBRO), whose operational early-warning system uses live rain-gauge networks. This model's output must never be presented as operationally or legally verified, and it is intended to augment — not replace — official hazard-zonation maps produced by field geology and geotechnical survey.

## Post-event labels used for a susceptibility task, not circular detection

The ACCIMT inventory records landslide scars mapped *after* Cyclone Ditwah — it is a damage-assessment dataset, by ACCIMT's own stated purpose. Using it to train a "forecasting" model is only valid if every model **input** (Sentinel-2 imagery, terrain derivatives) is drawn from strictly **pre-event** dates. If post-event imagery were used as input, the model would simply re-detect each scar's own already-visible signature (exposed soil, vegetation loss) — a circular result, not a forecast. Every training/eval patch in this pipeline is logged with its source imagery's acquisition date specifically so this constraint is independently auditable, not just assumed.

ACCIMT's own metadata supports the susceptibility framing: "the best predictor of future landslides is the location of past landslides... past landslide maps serve as reliable reference data for training and validating landslide prediction models."

## Minimum mappable unit

16.0% of the 4,225 labeled polygons have a minor-axis under 30 m, and 5.9% are under 20 m — at or below what Sentinel-2's 10–20 m bands can reliably resolve. This ceiling is not only sensor-imposed: the source labels themselves were geometrically smoothed at a 20 m tolerance (ArcGIS `SmoothPolygon`, PAEK algorithm) during ACCIMT's own processing, so sub-20 m boundary precision was never present in the ground truth to begin with.

Polygons below a 30 m minor-axis are excluded from the primary training target and instead reported as a separate "small-object" evaluation cohort, so recall on this cohort is measured and disclosed rather than silently absorbed into (or silently dropped from) the headline metrics. *(Actual measured recall on this cohort will be added here once evaluation runs — this section will be updated, not treated as complete, when that happens.)*

## Single-event, single-region generalization

Every positive label comes from one storm (Cyclone Ditwah) in one country. Spatial cross-validation (clustered + buffered, never random k-fold) gives an honest estimate of generalization to *unseen locations within this dataset's extent* — it cannot and does not estimate generalization to a different storm, season, or region. That limitation holds regardless of cross-validation design and is stated here rather than implied by a high validation score.

## Usage restriction

Per ACCIMT's data disclaimer (full text in `docs/data_card.md`): research/academic/planning use only, no guarantee of accuracy, independent verification required before any operational, legal, or commercial use, and ACCIMT must be credited in any derived output.

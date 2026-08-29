# AI Tool Usage Declaration

This project used **Claude Code** (Anthropic) as a coding assistant throughout development of `notebooks/landslide_pipeline.ipynb`, its `configs/*.yaml`, and the `docs/*.md` files in this repository. Declared here proactively, per the assignment's AI-use governance rules (Minimum Deliverable #17) and its explicit penalty range for concealing external assistance or presenting AI-assisted decisions as independently reasoned.

## What it was used for

- **Implementation**: writing and debugging the acquisition (Sentinel-2/DEM via CDSE openEO), preprocessing (cloud masking, spectral indices, terrain derivatives, patch extraction), sampling (slope-stratified negatives, hard negatives, spatial clustering, buffered cross-validation), modeling (U-Net/DeepLabV3+, Focal+Dice loss), and evaluation (segmentation metrics, held-out spatial-CV loop, ROC-AUC) code in the notebook.
- **Debugging**: diagnosing and fixing real runtime failures encountered while running this pipeline against live data and a live CDSE API — memory errors from full-scene array operations at real AOI scale, a stale-tempfile bug in two self-checks, a GPU-device-never-used bug (the model trained entirely on CPU despite a GPU being available), an HTTP 416 range-request rejection from the acquisition backend, and others.
- **Documentation review**: auditing the notebook's own markdown cells and `docs/*.md` for claims that had gone stale relative to the actual code (e.g., a description of a data-loading step that no longer matched its rewritten implementation).
- **Design discussion**: comparing candidate architectures and improvement proposals (rainfall features, ensembling, Attention U-Net) against the assignment brief and this project's own locked scope decisions, and reconciling the project plan against the actual assignment requirements once the brief was available.

## What it was not used for

- Fabricating results, citations, or literature claims. `docs/literature_survey.md`'s sources were retrieved via real web search, not recalled from model memory, and are marked as such.
- Deciding the project's scientific framing unilaterally. The susceptibility-vs-forecasting distinction, the pre-event-only constraint, and the minimum-mappable-unit threshold were derived from directly inspecting the actual ACCIMT dataset and its metadata (`docs/data_card.md`), not asserted from general knowledge.

## Team responsibility

Every component listed above is expected to be understood and defensible by the team without AI assistance present, per the assignment's individual-evaluation requirement. Non-obvious design decisions are documented with their rationale in `docs/architecture.md` and in the relevant function's docstring in the notebook — not left implicit — specifically so they can be explained and defended independently of how they were first implemented.

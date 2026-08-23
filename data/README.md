# data/ (gitignored — nothing here is committed except this file)

## 1. Labels (ACCIMT landslide inventory)

Copy the shapefile set (`.shp .shx .dbf .prj .cpg`, at minimum) to:

    data/raw/labels/Lanslides_Ditwah_2025.shp   (+ sibling files)

Source: ACCIMT Cyclone Ditwah landslide inventory — see `../docs/data_card.md`
for provenance, credit, and the required disclaimer. Not redistributed in
this repo; get it from the course-provided source.

## 2. Sentinel-2 / DEM

Set `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` (Colab Secrets, or `.env` locally — see `../.env.example`), then run the Step 2 acquisition cells in `../notebooks/landslide_pipeline.ipynb`.

This populates:

    data/raw/sentinel2/
    data/raw/dem/

(In Colab, "data/" means the Drive-backed `DATA_DIR` the notebook's setup cell prints, not a local folder.) Both downloads read AOI/date/band/cloud settings from `../configs/acquisition.yaml` — edit that, not the notebook, to change the search window or cloud threshold.

## 3. Processed patches (Step 3)

`data/processed/manifest.csv` will record, per patch: patch id, source Sentinel-2 product id, sensing date, is-positive flag, and CV cluster assignment — the audit trail that makes "pre-event-only" compliance (see `../docs/limitations.md`) independently checkable rather than assumed. `manifest_row()` + `save_manifest()` are demonstrated end-to-end (accumulate rows, write CSV) against one demo row at `data/processed/manifest_demo.csv`; writing the real manifest happens once real patches are generated against downloaded imagery.

Everything under `data/` is derived — regenerating it from `raw/` is always safe, nothing here is hand-edited.

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

`data/processed/manifest.csv` records, per patch: patch id, source Sentinel-2 product id, sensing date, and CV cluster assignment — the audit trail that makes "pre-event-only" compliance (see `../docs/limitations.md`) independently checkable rather than assumed. The notebook's `manifest_row()` builds one row at a time; writing the full manifest happens once real patches are generated against downloaded imagery (not yet run end-to-end — see the notebook's Status section).

Everything under `data/` is derived — regenerating it from `raw/` is always safe, nothing here is hand-edited.

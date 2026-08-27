# Landslide Forecasting System

Vision-based landslide susceptibility mapping for Sri Lanka, built on the ACCIMT Cyclone Ditwah (Nov 2025) landslide inventory.

## Project framing

This project predicts landslide **susceptibility** from pre-event Sentinel-2 imagery and DEM-derived terrain (slope, aspect, curvature — TWI is a documented future addition, not yet implemented, see [docs/limitations.md](docs/limitations.md)). It is not a real-time forecast and not a certified early-warning system.

## Data & service credit

Landslide inventory prepared by Mahesh Chathurange and W.G.N.N Jayawardhana (Research Scientists, Space Applications Division) on behalf of the **Arthur C. Clarke Institute for Modern Technologies (ACCIMT)**. Full dataset description, disclaimer, and usage terms: [docs/data_card.md](docs/data_card.md).

Satellite imagery: Copernicus Sentinel-2 L2A and Copernicus DEM (GLO-30), accessed via the **Copernicus Data Space Ecosystem (CDSE)** through its `openeo` Python client. Contains modified Copernicus Sentinel data, processed by ESA/CDSE.

## Status

Steps 1–6 implemented in [`notebooks/landslide_pipeline.ipynb`](notebooks/landslide_pipeline.ipynb): configuration & label loading, data acquisition & preprocessing, sample generation & spatial cross-validation, modeling, evaluation, and write-up notes. Sentinel-2 and DEM acquisition are proven against the live CDSE API at pilot scale; full-country-scale acquisition and real model training are next, intended to run in Colab. Everything downstream of real pixels is correctly implemented and self-checked against synthetic data pending that run. See [docs/report.md](docs/report.md) for the consolidated write-up (methodology, critical self-review, current status) or [docs/architecture.md](docs/architecture.md) for per-decision rationale.

## Setup

**Colab (recommended — for compute):** open [`notebooks/landslide_pipeline.ipynb`](notebooks/landslide_pipeline.ipynb) in Colab (File > Open notebook > GitHub, this repo). The first cell mounts Drive, clones the repo, and installs dependencies. Add `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` under Colab's Secrets (key icon) before running the acquisition cells.

**Local:**
```
python -m venv .venv
.venv\Scripts\activate       # Windows; use `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
cp .env.example .env         # fill in Copernicus Data Space Ecosystem credentials
```
Then open `notebooks/landslide_pipeline.ipynb` with that environment as the kernel.

## Repository layout

- `notebooks/landslide_pipeline.ipynb` — the pipeline: configuration, label loading, acquisition, preprocessing, sample generation, spatial cross-validation (Steps 1–3)
- `configs/` — YAML configuration read by the notebook (AOI, dates, bands, patch size, CV settings)
- `docs/` — data card, limitations, architecture notes, literature survey, ablation study design, consolidated report
- `data/` — gitignored; see `data/README.md` to regenerate locally

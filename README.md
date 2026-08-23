# Landslide Forecasting System

Vision-based landslide susceptibility mapping for Sri Lanka, built on the ACCIMT Cyclone Ditwah (Nov 2025) landslide inventory.

## Project framing

This project predicts landslide **susceptibility** from pre-event Sentinel-2 imagery and terrain data (slope, aspect, TWI). It is not a real-time forecast and not a certified early-warning system — see [docs/limitations.md](docs/limitations.md) before drawing conclusions from any output.

## Data credit

Landslide inventory prepared by Mahesh Chathurange and W.G.N.N Jayawardhana (Research Scientists, Space Applications Division) on behalf of the **Arthur C. Clarke Institute for Modern Technologies (ACCIMT)**. Full dataset description, disclaimer, and usage terms: [docs/data_card.md](docs/data_card.md).

## Status

Steps 1–3 implemented in [`notebooks/landslide_pipeline.ipynb`](notebooks/landslide_pipeline.ipynb): configuration & label loading, data acquisition & preprocessing, sample generation & spatial cross-validation. Every non-trivial function in it is checked inline against synthetic data. Modeling, evaluation, and write-up are later milestones, not started.

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
- `docs/` — data card, limitations, architecture notes
- `data/` — gitignored; see `data/README.md` to regenerate locally

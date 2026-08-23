# Landslide Forecasting System

Vision-based landslide susceptibility mapping for Sri Lanka, built on the ACCIMT Cyclone Ditwah (Nov 2025) landslide inventory.

## Project framing

This project predicts landslide **susceptibility** from pre-event Sentinel-2 imagery and terrain data (slope, aspect, TWI). It is not a real-time forecast and not a certified early-warning system — see [docs/limitations.md](docs/limitations.md) before drawing conclusions from any output.

## Data credit

Landslide inventory prepared by Mahesh Chathurange and W.G.N.N Jayawardhana (Research Scientists, Space Applications Division) on behalf of the **Arthur C. Clarke Institute for Modern Technologies (ACCIMT)**. Full dataset description, disclaimer, and usage terms: [docs/data_card.md](docs/data_card.md).

## Status

Early development. Current scope: data pipeline and sample-generation/spatial-cross-validation groundwork. Modeling has not started.

## Setup

```
conda env create -f environment.yml
conda activate landslide-forecasting
cp .env.example .env  # fill in Copernicus Data Space Ecosystem credentials
```

## Repository layout

- `src/acquisition/` — satellite imagery, DEM, and label loading
- `src/preprocessing/` — cloud masking, spectral indices, rasterization, patch extraction
- `src/datasets/` — positive/negative sample generation
- `src/training/` — spatial cross-validation fold generation
- `tests/` — unit tests (synthetic fixtures only, never real data)
- `docs/` — data card, limitations, architecture notes
- `data/` — gitignored; see `data/README.md` (added in the data-pipeline stage) to regenerate locally

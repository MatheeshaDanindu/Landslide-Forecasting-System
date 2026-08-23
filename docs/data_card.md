# Data Card — ACCIMT Cyclone Ditwah Landslide Inventory

## Source & credit

Prepared by Mahesh Chathurange and W.G.N.N Jayawardhana (Research Scientists, Space Applications Division), on behalf of the **Arthur C. Clarke Institute for Modern Technologies (ACCIMT)**, Sri Lanka's national mandated institute for space science activities.

Mapped from Sentinel-2 satellite imagery following Cyclone Ditwah (landfall Nov 28, 2025), using remote sensing and GIS techniques, in support of disaster risk reduction and land management.

## Official disclaimer (ACCIMT, verbatim from source metadata)

> This landslide boundary demarcation dataset related to Cyclone Ditwah – 2025 has been prepared by the Arthur C. Clarke Institute for Modern Technologies (ACCIMT) using available satellite data and standard analytical methods. While every effort has been made to ensure the accuracy and reliability of the information, ACCIMT makes no guarantee, express or implied, regarding the completeness, accuracy, or suitability of the data for any specific purpose. The data is provided for research, academic, planning, and decision-support purposes only. ACCIMT shall not be held responsible for any errors, omissions, or consequences arising from the use, interpretation, or application of this dataset. Users are advised to independently verify the data before using it for operational, legal, or commercial purposes. Any use of this dataset should appropriately acknowledge the Arthur C. Clarke Institute for Modern Technologies (ACCIMT).

## Dataset facts (verified directly, not from secondary description)

- **4,225 unique polygons**, no duplicate IDs. Geometry: `Polygon`/`MultiGeometry`, no points or lines.
- **CRS:** WGS84 geographic (EPSG:4326) in both the delivered `.kmz` and `.shp` exports; reprojected to UTM Zone 44N (EPSG:32644) for this project's metric work (patch sizing, buffers).
- **Extent:** lon [80.02°, 81.40°], lat [6.56°, 8.07°] — roughly 150 km × 165 km across Sri Lanka's central highlands. Spans multiple Sentinel-2 MGRS tiles.
- **Size distribution** (per-polygon bounding box): width median 84 m (p90 260 m, p99 917 m, max 3,552 m); area median 3,067 m² (max 1.29 km²). Minor-axis (narrower side) median 63 m; **16.0% of polygons have a minor axis under 30 m, 5.9% under 20 m** — near or below Sentinel-2's 10–20 m band resolution. See [limitations.md](limitations.md) for how this is handled.
- **Attributes are minimal.** No per-polygon district, date, or severity field exists in either delivered format. The `.kmz`'s embedded HTML description carries only `FID`/`Id`/`ClassName` (72% blank, 28% `"change"`); the `.shp`'s `.dbf` carries only a non-sequential legacy `Id` integer. Any spatial grouping (e.g. for cross-validation) is computed by this project, not read from the source data.
- **Processing lineage** (from the shapefile's ArcGIS metadata): boundaries were digitized, then generalized with the `SmoothPolygon` tool (PAEK algorithm, 20 m tolerance), then merged from two source batches (`Missed_smoothed` + `final_landslides_smoothed`). Sub-20 m boundary precision was not preserved by this process.
- **Stated purpose** (ACCIMT's own metadata): primarily post-event damage assessment and disaster response planning. ACCIMT's metadata also explicitly notes that past-landslide locations are a recognized reference for training/validating landslide prediction models — the basis for this project's susceptibility-mapping framing (see [limitations.md](limitations.md)).

## Local storage

The raw `.kmz`/`.shp` files are kept outside this repository (see `data/README.md`, added in the data-pipeline stage, for the exact local path and regeneration steps) and are gitignored — never commit them.

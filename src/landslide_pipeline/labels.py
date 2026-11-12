from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np


def _oriented_minor_axis(geom):
    try:
        return geom.minimum_rotated_rectangle.length / 2.0
    except Exception:
        return np.nan


def load_labels(path: str | Path, crs: str = "EPSG:32644") -> gpd.GeoDataFrame:
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Landslide labels not found: {label_path}")

    gdf = gpd.read_file(label_path)
    if gdf.empty:
        raise ValueError(f"No landslide geometries loaded from {label_path}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string().upper() != crs.upper():
        gdf = gdf.to_crs(crs)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf["minor_axis_m"] = gdf.geometry.apply(_oriented_minor_axis)
    return gdf


def record_label_summary(labels: gpd.GeoDataFrame) -> Dict[str, Any]:
    data = labels.copy()
    total = len(data)
    return {
        "total_polygons": int(total),
        "total_area_m2": float(data["area_m2"].sum()),
        "mean_area_m2": float(data["area_m2"].mean()),
        "small_objects": int((data["minor_axis_m"] < 30).sum()),
        "small_share": float((data["minor_axis_m"] < 30).mean()),
        "crs": data.crs.to_string() if data.crs else "unknown",
    }

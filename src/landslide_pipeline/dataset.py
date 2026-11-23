"""Standalone real-data dataset: label loading, spatial clustering, lazy
per-patch raster reads (no full-tile load), feature stack, and validity mask.

Consolidates logic proven out in the fold-0 evaluation script into the
package, per audit Issue 1 (no standalone pipeline) and Issue 4 (cloud mask
previously discarded in `extract_patch` -> `LandslideDataset.__getitem__`:
`x, _ = extract_patch(...)`). Here the mask is returned and used, not dropped.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box as sbox

from .labels import _oriented_minor_axis

BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# openEO Sentinel-2 L2A export order for this acquisition: bands + SCL last.
TILE_BAND_ORDER = BAND_ORDER + ["SCL"]


def load_labels(shp_path, metric_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string().upper() != metric_crs.upper():
        gdf = gdf.to_crs(metric_crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf["minor_axis_m"] = gdf.geometry.apply(_oriented_minor_axis)
    gdf["polygon_id"] = np.arange(len(gdf))
    return gdf


def assign_clusters(labels: gpd.GeoDataFrame, n_clusters: int, seed: int) -> gpd.GeoDataFrame:
    from sklearn.cluster import KMeans
    xy = np.column_stack([labels.geometry.centroid.x, labels.geometry.centroid.y])
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    out = labels.copy()
    out["cluster"] = km.fit_predict(xy)
    return out


def _ratio(a, b, eps=1e-6):
    b = np.asarray(b, np.float32)
    return np.where(np.abs(b) > eps, a / b, 0.0).astype(np.float32)


def _slope_aspect(dem, res):
    dy, dx = np.gradient(dem.astype(np.float32), res)
    return np.degrees(np.arctan(np.hypot(dx, dy))), np.mod(np.degrees(np.arctan2(dy, dx)), 360.0)


def _curvature(dem, res):
    dy, dx = np.gradient(dem.astype(np.float32), res)
    d2y, _ = np.gradient(dy, res)
    _, d2x = np.gradient(dx, res)
    return ((d2y + d2x) / 2.0).astype(np.float32)


def centroid_window_rc(cx, cy, transform, patch_size):
    col, row = ~transform * (cx, cy)
    half = patch_size // 2
    return math.floor(row) - half, math.floor(col) - half


def rasterize_polygons(polys: gpd.GeoDataFrame, transform, shape) -> np.ndarray:
    if len(polys) == 0:
        return np.zeros(shape, np.uint8)
    geoms = [(g, 1) for g in polys.geometry]
    return rasterize(geoms, out_shape=shape, transform=transform, fill=0,
                      dtype=np.uint8, all_touched=True)


def build_patch(s2_ds, dem_ds, row: int, col: int, label_mask_full: np.ndarray,
                 patch_size: int, resolution_m: float, reflectance_scale: float,
                 scl_exclude: set) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    """Reads one (16, P, P) feature patch + its (P, P) label + (P, P) validity
    mask (True = usable, non-cloud/shadow/nodata pixel) via windowed rasterio
    reads. Returns (None, None, None) if the window falls outside the raster
    or any read comes back the wrong shape."""
    margin = 2
    H = W = patch_size
    r0, c0 = max(0, row - margin), max(0, col - margin)
    r1 = min(s2_ds.height, row + H + margin)
    c1 = min(s2_ds.width, col + W + margin)
    if row < 0 or col < 0 or row + H > s2_ds.height or col + W > s2_ds.width:
        return None, None, None
    win = Window(c0, r0, c1 - c0, r1 - r0)
    data = s2_ds.read(window=win).astype(np.float32)
    n = data.shape[0]
    bd = {TILE_BAND_ORDER[i]: data[i] for i in range(min(n, len(TILE_BAND_ORDER)))}
    if "SCL" not in bd:
        return None, None, None
    dr, dc = row - r0, col - c0

    def crop(arr):
        return arr[dr:dr + H, dc:dc + W]

    scl = crop(bd["SCL"])
    valid = ~np.isin(scl, list(scl_exclude))

    win_d = Window(c0, r0, c1 - c0, r1 - r0)
    dem_patch = dem_ds.read(1, window=win_d).astype(np.float32)
    # Issue 20: DEM nodata was previously silently turned into 0 elevation (false
    # flat terrain) by reprojection code this pipeline doesn't use; here we mask
    # it explicitly instead, at both the feature and the validity-mask level.
    dem_nodata = dem_ds.nodata
    if dem_nodata is not None:
        dem_invalid = crop(dem_patch) == dem_nodata
        dem_patch = np.where(dem_patch == dem_nodata, np.nan, dem_patch)
    else:
        dem_invalid = np.zeros((H, W), dtype=bool)
    dem_patch = np.nan_to_num(dem_patch, nan=0.0)  # finite-difference needs no-NaN input
    slope, aspect = _slope_aspect(dem_patch, resolution_m)
    curv = _curvature(dem_patch, resolution_m)
    valid = valid & ~dem_invalid

    chs = [crop(bd[b]) * reflectance_scale for b in BAND_ORDER]
    r_, n_, bl_, g_, s_ = chs[2], chs[6], chs[0], chs[1], chs[8]
    chs.append(_ratio(n_ - r_, n_ + r_))
    chs.append(_ratio((s_ + r_) - (n_ + bl_), (s_ + r_) + (n_ + bl_)))
    chs.append(_ratio(g_ - s_, g_ + s_))
    chs.append(crop(slope))
    chs.append(crop(aspect))
    chs.append(crop(curv))
    stack = np.stack(chs, 0).astype(np.float32)
    if stack.shape != (16, H, W):
        return None, None, None

    y = label_mask_full[row:row + H, col:col + W]
    if y.shape != (H, W):
        return None, None, None
    return stack, y.astype(np.float32), valid.astype(np.float32)


def sample_negative_centers(dem_ds, s2_ds, positive_geoms, exclusion_buffer_m: float,
                             n_total: int, bins: int, rng: np.random.Generator):
    """Slope-stratified negative centers within s2_ds's extent, outside
    `exclusion_buffer_m` of every positive polygon. Matches the positive
    set's slope distribution (via quantile binning) instead of a uniform
    random background, which would be trivially separable by slope alone.
    Returns a list of (row, col) pixel centers.
    """
    if len(positive_geoms) == 0:
        return []
    exclusion = positive_geoms.buffer(exclusion_buffer_m).union_all()
    box = tile_box(s2_ds)
    left, bottom, right, top = box.bounds

    dem = dem_ds.read(1).astype(np.float32)
    nodata = dem_ds.nodata
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    slope, _ = _slope_aspect(np.nan_to_num(dem, nan=0.0), dem_ds.res[0])
    finite_slope = slope[np.isfinite(slope)]
    if finite_slope.size == 0:
        return []
    bin_edges = np.quantile(finite_slope, np.linspace(0, 1, bins + 1))

    n_try = max(n_total * 30, 200)
    xs = rng.uniform(left, right, n_try)
    ys = rng.uniform(bottom, top, n_try)
    from shapely.geometry import Point
    candidates = []
    for x, y in zip(xs, ys):
        if exclusion.contains(Point(x, y)):
            continue
        col, row = ~s2_ds.transform * (x, y)
        row, col = int(row), int(col)
        if row < 0 or col < 0 or row >= slope.shape[0] or col >= slope.shape[1]:
            continue
        s = slope[row, col]
        if not np.isfinite(s):
            continue
        candidates.append((row, col, s))

    per_bin = max(1, n_total // bins)
    out = []
    for b in range(bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = [c for c in candidates if lo <= c[2] <= hi]
        rng.shuffle(in_bin)
        out.extend([(r, c) for r, c, _ in in_bin[:per_bin]])
    return out


def tile_box(s2_ds):
    left, bottom, right, top = rasterio.transform.array_bounds(s2_ds.height, s2_ds.width, s2_ds.transform)
    return sbox(left, bottom, right, top)


class PatchDataset(torch.utils.data.Dataset):
    """In-memory dataset of already-extracted (x, y, valid_mask) triples --
    fine at the patch counts this pipeline samples (hundreds-low thousands);
    swap for lazy per-item raster reads if that stops being true."""

    def __init__(self, items: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, m = self.items[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)

from __future__ import annotations

import numpy as np


def valid_pixel_mask(arr: np.ndarray, invalid_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.isfinite(arr) & (arr != invalid_value)


def to_reflectance(array: np.ndarray, scale: float = 0.0001) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    return np.clip(arr * scale, 0.0, 1.0)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = np.asarray(denominator, dtype=np.float32)
    numer = np.asarray(numerator, dtype=np.float32)
    return np.divide(numer, denom + eps, out=np.zeros_like(numer, dtype=np.float32), where=np.abs(denom) > eps)


def ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return safe_ratio(nir - red, nir + red, eps)


def bsi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray, swir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    numerator = (swir + red) - (nir + blue)
    denominator = (swir + red) + (nir + blue)
    return safe_ratio(numerator, denominator, eps)


def mndwi(green: np.ndarray, swir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return safe_ratio(green - swir, green + swir, eps)


def slope_from_dem(dem: np.ndarray, pixel_size_m: float = 10.0) -> np.ndarray:
    dem = np.asarray(dem, dtype=np.float32)
    dx = np.gradient(dem, axis=1) / pixel_size_m
    dy = np.gradient(dem, axis=0) / pixel_size_m
    slope_rad = np.arctan(np.hypot(dx, dy))
    return np.degrees(slope_rad)


def aspect_from_dem(dem: np.ndarray, pixel_size_m: float = 10.0) -> np.ndarray:
    dem = np.asarray(dem, dtype=np.float32)
    dzdy = np.gradient(dem, axis=0) / pixel_size_m
    dzdx = np.gradient(dem, axis=1) / pixel_size_m
    return np.mod(np.degrees(np.arctan2(dzdy, dzdx)), 360.0)


def curvature_from_dem(dem: np.ndarray, pixel_size_m: float = 10.0) -> np.ndarray:
    dem = np.asarray(dem, dtype=np.float32)
    d2y = np.diff(dem, n=2, axis=0, prepend=dem[0:1]) / (pixel_size_m ** 2)
    d2x = np.diff(dem, n=2, axis=1, prepend=dem[:, 0:1]) / (pixel_size_m ** 2)
    return (d2y + d2x) / 2.0


def compose_16_channel_input(
    spectral_bands: dict,
    dem: np.ndarray,
    scale: float = 0.0001,
    pixel_size_m: float = 10.0,
) -> np.ndarray:
    """Compose the expected 16-channel feature tensor per 64x64 patch."""
    band_order = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    channels = []

    for key in band_order:
        band = np.asarray(spectral_bands[key], dtype=np.float32)
        channels.append(to_reflectance(band, scale))

    red = channels[2]
    nir = channels[6]
    blue = channels[0]
    green = channels[1]
    swir = channels[8]

    channels.append(ndvi(red, nir))
    channels.append(bsi(red, nir, blue, swir))
    channels.append(mndwi(green, swir))

    slope = slope_from_dem(dem, pixel_size_m)
    aspect = aspect_from_dem(dem, pixel_size_m)
    curvature = curvature_from_dem(dem, pixel_size_m)
    channels.extend([slope, aspect, curvature])

    if len(channels) != 16:
        raise ValueError(f"Expected 16 channels for the feature stack, got {len(channels)}")
    return np.stack(channels, axis=0).astype(np.float32)

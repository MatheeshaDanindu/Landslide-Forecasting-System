"""Evaluate a trained fold checkpoint on its held-out test cluster.

Metrics are computed with the cloud/shadow/nodata validity mask applied
(Issue 4) and include PR-AUC alongside ROC-AUC (Issue 12). Threshold is
frozen at 0.5 by default; pass --threshold-sweep to also report the
validation-selected operating point (Issue 11) if a *_history.csv exists.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from .config import load_project_config
from .dataset import (assign_clusters, build_patch, centroid_window_rc,
                       load_labels, rasterize_polygons, tile_box)
from .metrics import segmentation_metrics
from .models import build_model_for_key

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "experiments" / "corrected_experiment_v1" / "checkpoints"
RESULTS_PATH = REPO_ROOT / "experiments" / "corrected_experiment_v1" / "fold_metrics.csv"


def extract_test_patches(cfg, test_poly):
    prep = cfg["preprocessing"]
    patch_size = prep["patch"]["size_px"]
    res_m = prep["patch"]["resolution_m"]
    rscale = prep["reflectance_scale"]
    scl_exclude = set(prep["cloud_mask"]["scl_exclude"])

    tile_dirs = sorted((REPO_ROOT / "raw" / "sentinel2").glob("tile*"))
    dem_tiles = sorted((REPO_ROOT / "raw" / "dem").glob("dem_tile*.tif"))
    out = []
    for ti, tile_dir in enumerate(tile_dirs):
        tifs = sorted(tile_dir.glob("openEO_*.tif"))
        if not tifs or ti >= len(dem_tiles):
            continue
        with rasterio.open(tifs[0]) as s2_ds, rasterio.open(dem_tiles[ti]) as dem_ds:
            box = tile_box(s2_ds)
            in_tile = test_poly[test_poly.geometry.centroid.within(box)]
            if len(in_tile) == 0:
                continue
            label_full = rasterize_polygons(in_tile, s2_ds.transform, (s2_ds.height, s2_ds.width))
            for _, row_data in in_tile.iterrows():
                cx, cy = row_data.geometry.centroid.x, row_data.geometry.centroid.y
                r, c = centroid_window_rc(cx, cy, s2_ds.transform, patch_size)
                x, y, m = build_patch(s2_ds, dem_ds, r, c, label_full, patch_size, res_m, rscale, scl_exclude)
                if x is not None:
                    out.append((x, y, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model-key", choices=["baseline", "comparison"], default="baseline")
    ap.add_argument("--threshold", type=float, default=None,
                     help="defaults to the fold's validation-selected threshold if a "
                          "*_meta.json exists, else 0.5")
    args = ap.parse_args()

    cfg = load_project_config(REPO_ROOT)
    prep, spatial = cfg["preprocessing"], cfg["spatial_cv"]
    min_minor = prep["labels"]["min_minor_axis_m"]

    # acquisition.yaml's labels.path is relative to DATA_DIR (= REPO_ROOT/"data"),
    # not REPO_ROOT -- see docs/architecture.md DATA_DIR note; the Sentinel-2/DEM
    # tiles below are, inconsistently, directly under REPO_ROOT/"raw" (flagged in
    # AUDIT_REPORT.md Issue 22).
    labels = assign_clusters(load_labels(REPO_ROOT / "data" / cfg["acquisition"]["labels"]["path"],
                                          prep["crs"]["metric"]),
                              spatial["clustering"]["n_clusters"], spatial["random_seed"])
    primary = labels[labels["minor_axis_m"] >= min_minor]
    test_poly = primary[primary["cluster"] == args.fold]
    patches = extract_test_patches(cfg, test_poly)
    print(f"{len(patches)} test patches for fold {args.fold}")
    if not patches:
        print("No test patches -- aborting.")
        return

    ckpt_path = CHECKPOINT_ROOT / f"fold{args.fold}_{args.model_key}.pt"
    threshold = args.threshold
    meta_path = CHECKPOINT_ROOT / f"fold{args.fold}_{args.model_key}_meta.json"
    if threshold is None:
        if meta_path.exists():
            threshold = json.loads(meta_path.read_text())["selected_threshold"]
            print(f"Using validation-selected threshold {threshold:.2f} from {meta_path.name}")
        else:
            threshold = 0.5
    args.threshold = threshold
    if not ckpt_path.exists():
        # fall back to the pre-existing repo checkpoint for fold 0, so this
        # script also works before any corrected retrain has been run
        legacy = {"baseline": "fold_cluster0_baseline.pt", "comparison": "fold_cluster0_comparison.pt"}
        ckpt_path = REPO_ROOT / "Trained Model" / legacy[args.model_key]
        print(f"No corrected checkpoint found; evaluating legacy checkpoint {ckpt_path.name}")

    device = torch.device("cpu")
    model = build_model_for_key(cfg, args.model_key)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval().to(device)

    all_y, all_p, all_m = [], [], []
    bs = 32
    for i in range(0, len(patches), bs):
        batch = patches[i:i + bs]
        x = torch.tensor(np.stack([b[0] for b in batch])).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()
        for j, (_, y, m) in enumerate(batch):
            all_y.append(y.ravel())
            all_p.append(probs[j, 0].ravel())
            all_m.append(m.ravel().astype(bool))

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    valid = np.concatenate(all_m)
    print(f"valid pixels: {valid.sum()}/{valid.size} ({valid.mean():.1%}) after cloud/shadow/nodata mask")

    metrics = segmentation_metrics(y_true, y_pred, threshold=args.threshold, valid_mask=valid)
    metrics.update({"fold": args.fold, "model_key": args.model_key, "threshold": args.threshold,
                    "n_patches": len(patches), "n_valid_pixels": int(valid.sum()), "n_total_pixels": int(valid.size)})
    print(json.dumps(metrics, indent=2))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            w.writeheader()
        w.writerow(metrics)
    print(f"Appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

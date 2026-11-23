"""Standalone, config-driven training for one spatial fold.

Fixes applied relative to the notebook path (see repo AUDIT_REPORT.md):
  - minor-axis filtering uses the corrected shortest-edge calc (labels.py)
  - cloud/shadow/nodata validity mask is propagated into the loss (masked
    focal-dice), not discarded
  - inner train/val split is grouped by polygon_id, not a random patch split
    (torch.utils.data.random_split over individual patches could put two
    patches of the same polygon on both sides)
  - best-validation checkpoint is kept, not just the last epoch
  - early stopping on inner-val loss

Usage:
    PYTHONPATH=src python -m landslide_pipeline.train --fold 0 --model-key baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

from .config import load_project_config
from .dataset import (PatchDataset, assign_clusters, build_patch, load_labels,
                       tile_box)
from .models import build_model_for_key
from .training import focal_dice_loss, seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "experiments" / "corrected_experiment_v1" / "checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="held-out cluster id (0-7)")
    ap.add_argument("--model-key", choices=["baseline", "comparison"], default="baseline")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-train-polygons", type=int, default=None,
                     help="cap for a fast smoke run; omit for the full fold")
    args = ap.parse_args()

    cfg = load_project_config(REPO_ROOT)
    prep, spatial, model_cfg = cfg["preprocessing"], cfg["spatial_cv"], cfg["model"]
    seed = spatial["random_seed"]
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    patch_size = prep["patch"]["size_px"]
    res_m = prep["patch"]["resolution_m"]
    rscale = prep["reflectance_scale"]
    scl_exclude = set(prep["cloud_mask"]["scl_exclude"])
    min_minor = prep["labels"]["min_minor_axis_m"]
    buffer_m = spatial["buffer_m"]

    # labels.path is DATA_DIR-relative; the raw S2/DEM tiles are REPO_ROOT-relative
    # -- a real path inconsistency, see AUDIT_REPORT.md Issue 22.
    labels = assign_clusters(load_labels(REPO_ROOT / "data" / cfg["acquisition"]["labels"]["path"],
                                          prep["crs"]["metric"]),
                              spatial["clustering"]["n_clusters"], seed)
    primary = labels[labels["minor_axis_m"] >= min_minor]
    test_poly = primary[primary["cluster"] == args.fold]
    test_buffer = test_poly.geometry.buffer(buffer_m).union_all()
    train_poly = primary[(primary["cluster"] != args.fold) & (~primary.geometry.within(test_buffer))]
    if args.max_train_polygons:
        train_poly = train_poly.iloc[:args.max_train_polygons]

    # grouped inner split by polygon_id -- never splits a polygon's own patch
    poly_ids = train_poly["polygon_id"].to_numpy()
    rng.shuffle(poly_ids)
    n_val = max(1, int(0.15 * len(poly_ids)))
    val_ids, fit_ids = set(poly_ids[:n_val]), set(poly_ids[n_val:])

    tile_dirs = sorted((REPO_ROOT / "raw" / "sentinel2").glob("tile*"))
    dem_tiles = sorted((REPO_ROOT / "raw" / "dem").glob("dem_tile*.tif"))

    def extract(poly_df):
        out = []
        for ti, tile_dir in enumerate(tile_dirs):
            tifs = sorted(tile_dir.glob("openEO_*.tif"))
            if not tifs or ti >= len(dem_tiles):
                continue
            with rasterio.open(tifs[0]) as s2_ds, rasterio.open(dem_tiles[ti]) as dem_ds:
                box = tile_box(s2_ds)
                in_tile = poly_df[poly_df.geometry.centroid.within(box)]
                if len(in_tile) == 0:
                    continue
                label_full = __import__("landslide_pipeline.dataset", fromlist=["rasterize_polygons"]) \
                    .rasterize_polygons(in_tile, s2_ds.transform, (s2_ds.height, s2_ds.width))
                for _, row_data in in_tile.iterrows():
                    cx, cy = row_data.geometry.centroid.x, row_data.geometry.centroid.y
                    from .dataset import centroid_window_rc
                    r, c = centroid_window_rc(cx, cy, s2_ds.transform, patch_size)
                    x, y, m = build_patch(s2_ds, dem_ds, r, c, label_full, patch_size, res_m, rscale, scl_exclude)
                    if x is not None:
                        out.append((x, y, m, int(row_data["polygon_id"])))
        return out

    fit_patches = [t[:3] for t in extract(train_poly) if t[3] in fit_ids]
    val_patches = [t[:3] for t in extract(train_poly) if t[3] in val_ids]
    test_patches = [t[:3] for t in extract(test_poly)]

    print(f"fold={args.fold} model={args.model_key} "
          f"fit={len(fit_patches)} val={len(val_patches)} test={len(test_patches)} patches")
    if not fit_patches:
        print("No training patches assembled for this fold -- aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_key(cfg, args.model_key).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model_cfg["training"]["learning_rate"])
    loss_cfg = model_cfg["loss"]
    epochs = args.epochs or model_cfg["training"]["epochs"]
    batch_size = model_cfg["training"]["batch_size"]

    fit_loader = torch.utils.data.DataLoader(PatchDataset(fit_patches), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(PatchDataset(val_patches), batch_size=batch_size) if val_patches else None

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_ROOT / f"fold{args.fold}_{args.model_key}.pt"
    history_path = CHECKPOINT_ROOT / f"fold{args.fold}_{args.model_key}_history.csv"

    best_val = float("inf")
    bad_epochs = 0
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y, m in fit_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits = model(x).squeeze(1)
            loss = focal_dice_loss(logits, y, alpha=loss_cfg["focal_alpha"], gamma=loss_cfg["focal_gamma"],
                                    dice_weight=loss_cfg["dice_weight"], valid_mask=m)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        val_loss = float("nan")
        if val_loader:
            model.eval()
            vals = []
            with torch.no_grad():
                for x, y, m in val_loader:
                    x, y, m = x.to(device), y.to(device), m.to(device)
                    logits = model(x).squeeze(1)
                    vals.append(focal_dice_loss(logits, y, alpha=loss_cfg["focal_alpha"],
                                                 gamma=loss_cfg["focal_gamma"],
                                                 dice_weight=loss_cfg["dice_weight"], valid_mask=m).item())
            val_loss = float(np.mean(vals))

        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({time.time()-t0:.0f}s elapsed)")

        monitor = val_loss if val_loader else train_loss
        if monitor < best_val:
            best_val = monitor
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stopping at epoch {epoch} (patience={args.patience})")
                break

    with open(history_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(history)

    meta = {
        "fold": args.fold, "model_key": args.model_key, "arch": model_cfg[args.model_key]["arch"],
        "encoder": model_cfg[args.model_key]["encoder"], "in_channels": model_cfg["input_channels"],
        "seed": seed, "n_fit_patches": len(fit_patches), "n_val_patches": len(val_patches),
        "n_test_patches": len(test_patches), "best_val_loss": best_val, "epochs_run": len(history),
    }
    with open(CHECKPOINT_ROOT / f"fold{args.fold}_{args.model_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved best checkpoint to {ckpt_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

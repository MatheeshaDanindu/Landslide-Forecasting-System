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
from .dataset import (PatchDataset, assign_clusters, build_patch,
                       centroid_window_rc, load_labels, rasterize_polygons,
                       sample_negative_centers, tile_box)
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

    neg_cfg = spatial["negative_sampling"]

    def extract(poly_df, with_negatives=False):
        out, neg_out = [], []
        for ti, tile_dir in enumerate(tile_dirs):
            tifs = sorted(tile_dir.glob("openEO_*.tif"))
            if not tifs or ti >= len(dem_tiles):
                continue
            with rasterio.open(tifs[0]) as s2_ds, rasterio.open(dem_tiles[ti]) as dem_ds:
                box = tile_box(s2_ds)
                in_tile = poly_df[poly_df.geometry.centroid.within(box)]
                if len(in_tile) == 0:
                    continue
                label_full = rasterize_polygons(in_tile, s2_ds.transform, (s2_ds.height, s2_ds.width))
                for _, row_data in in_tile.iterrows():
                    cx, cy = row_data.geometry.centroid.x, row_data.geometry.centroid.y
                    r, c = centroid_window_rc(cx, cy, s2_ds.transform, patch_size)
                    x, y, m = build_patch(s2_ds, dem_ds, r, c, label_full, patch_size, res_m, rscale, scl_exclude)
                    if x is not None:
                        out.append((x, y, m, int(row_data["polygon_id"])))

                if with_negatives:
                    centers = sample_negative_centers(
                        dem_ds, s2_ds, in_tile.geometry, neg_cfg["exclusion_buffer_m"],
                        n_total=neg_cfg["slope_bins"] * neg_cfg["n_per_bin"],
                        bins=neg_cfg["slope_bins"], rng=rng)
                    empty_label = np.zeros((s2_ds.height, s2_ds.width), dtype=np.uint8)
                    half = patch_size // 2
                    for row, col in centers:
                        x, y, m = build_patch(s2_ds, dem_ds, row - half, col - half, empty_label,
                                               patch_size, res_m, rscale, scl_exclude)
                        if x is not None:
                            neg_out.append((x, y, m))
        return out, neg_out

    train_pos, train_neg = extract(train_poly, with_negatives=True)
    test_pos, _ = extract(test_poly, with_negatives=False)

    fit_patches = [t[:3] for t in train_pos if t[3] in fit_ids]
    val_patches = [t[:3] for t in train_pos if t[3] in val_ids]
    test_patches = [t[:3] for t in test_pos]

    # negatives have no polygon identity to group by -- an independent random
    # 85/15 split per sampled location carries no leakage risk the way a
    # patch-level split of positives would.
    rng.shuffle(train_neg)
    n_neg_val = max(1, int(0.15 * len(train_neg))) if train_neg else 0
    val_patches += train_neg[:n_neg_val]
    fit_patches += train_neg[n_neg_val:]

    print(f"fold={args.fold} model={args.model_key} "
          f"fit={len(fit_patches)} (neg={len(train_neg)-n_neg_val}) "
          f"val={len(val_patches)} test={len(test_patches)} patches")
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

    # Threshold selection on inner validation only, frozen before the outer
    # test cluster is ever touched (Issue 11) -- reload the best checkpoint,
    # sweep 0.05-0.95, pick the F1-maximizing threshold on val_patches.
    selected_threshold = 0.5
    if val_patches:
        from .metrics import segmentation_metrics
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        model.eval()
        all_y, all_p, all_m = [], [], []
        with torch.no_grad():
            for x, y, m in val_loader:
                probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
                for j in range(len(y)):
                    all_y.append(y[j].numpy().ravel())
                    all_p.append(probs[j, 0].ravel())
                    all_m.append(m[j].numpy().ravel().astype(bool))
        y_val, p_val, m_val = np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_m)
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.05, 0.96, 0.05):
            f1 = segmentation_metrics(y_val, p_val, threshold=float(t), valid_mask=m_val)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        selected_threshold = best_t
        print(f"  validation-selected threshold: {selected_threshold:.2f} (val F1={best_f1:.4f})")

    meta = {
        "fold": args.fold, "model_key": args.model_key, "arch": model_cfg[args.model_key]["arch"],
        "encoder": model_cfg[args.model_key]["encoder"], "in_channels": model_cfg["input_channels"],
        "seed": seed, "n_fit_patches": len(fit_patches), "n_val_patches": len(val_patches),
        "n_test_patches": len(test_patches), "best_val_loss": best_val, "epochs_run": len(history),
        "selected_threshold": selected_threshold,
    }
    with open(CHECKPOINT_ROOT / f"fold{args.fold}_{args.model_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved best checkpoint to {ckpt_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

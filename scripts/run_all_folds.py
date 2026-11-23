"""Driver: for each assigned fold, extract patches ONCE and train+evaluate
BOTH architectures from that single extraction (previously each model
re-ran the full rasterio/numpy extraction loop separately -- pure waste,
since both models train on identical patches for a given fold).

Run several of these as separate OS processes with --shard/--num-shards to
use multiple CPU cores in parallel -- same data, same hyperparameters, same
code path per fold, so there's no accuracy tradeoff, only wall-clock.
Extraction is a single-threaded Python/rasterio loop, so --threads only
affects the torch training step; it's still worth capping so N concurrent
processes don't oversubscribe the machine's cores.

    python scripts/run_all_folds.py --shard 0 --num-shards 8 --threads 1
    ... (one invocation per shard, launched concurrently)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from landslide_pipeline.config import load_project_config
from landslide_pipeline.metrics import segmentation_metrics
from landslide_pipeline.models import build_model_for_key
from landslide_pipeline.train import (REPO_ROOT, extract_fold_patches,
                                       train_one_model)
from landslide_pipeline.training import seed_everything

RESULTS_PATH = ROOT / "experiments" / "corrected_experiment_v1" / "fold_metrics.csv"
MODELS = ["baseline", "comparison"]


def eval_in_memory(cfg, fold, model_key, test_patches, meta):
    """Evaluate the just-trained (in-memory-checkpointed) model on test_patches
    without re-extracting anything -- the checkpoint was just written by
    train_one_model, so this just reloads weights and runs inference."""
    import csv

    device = torch.device("cpu")
    model = build_model_for_key(cfg, model_key)
    ckpt_path = ROOT / "experiments" / "corrected_experiment_v1" / "checkpoints" / f"fold{fold}_{model_key}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model.eval().to(device)

    all_y, all_p, all_m = [], [], []
    bs = 32
    for i in range(0, len(test_patches), bs):
        batch = test_patches[i:i + bs]
        x = torch.tensor(np.stack([b[0] for b in batch])).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()
        for j, (_, y, m) in enumerate(batch):
            all_y.append(y.ravel())
            all_p.append(probs[j, 0].ravel())
            all_m.append(m.ravel().astype(bool))
    if not all_y:
        print(f"fold={fold} model={model_key}: no test patches, skipping eval")
        return

    y_true, y_pred, valid = np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_m)
    threshold = meta["selected_threshold"]
    metrics = segmentation_metrics(y_true, y_pred, threshold=threshold, valid_mask=valid)
    metrics.update({"fold": fold, "model_key": model_key, "threshold": threshold,
                     "n_patches": len(test_patches), "n_valid_pixels": int(valid.sum()),
                     "n_total_pixels": int(valid.size)})
    print(f"fold={fold} model={model_key} RESULT: {json.dumps(metrics)}", flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            w.writeheader()
        w.writerow(metrics)


def run_fold(cfg, fold, seed, rng):
    t0 = time.time()
    print(f"\n=== EXTRACT fold={fold} started {time.strftime('%H:%M:%S')} ===", flush=True)
    fit_patches, val_patches, test_patches = extract_fold_patches(cfg, fold, seed, rng)
    print(f"=== EXTRACT fold={fold} done in {time.time()-t0:.0f}s: "
          f"fit={len(fit_patches)} val={len(val_patches)} test={len(test_patches)} ===", flush=True)

    for model_key in MODELS:
        t1 = time.time()
        meta = train_one_model(cfg, fold, model_key, fit_patches, val_patches, test_patches)
        print(f"=== TRAIN fold={fold} model={model_key} done in {time.time()-t1:.0f}s ===", flush=True)
        if meta:
            eval_in_memory(cfg, fold, model_key, test_patches, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_project_config(REPO_ROOT)
    seed = cfg["spatial_cv"]["random_seed"]

    my_folds = list(range(8))[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: folds {my_folds}", flush=True)

    for fold in my_folds:
        seed_everything(seed)  # re-seed per fold so each fold's own RNG use is reproducible in isolation
        rng = np.random.default_rng(seed + fold)  # distinct negative-sampling draw per fold, still deterministic
        try:
            run_fold(cfg, fold, seed, rng)
        except Exception:
            print(f"!!! fold={fold} FAILED, skipping !!!", flush=True)
            traceback.print_exc()
    print(f"\n=== SHARD {args.shard} DONE ===", flush=True)


if __name__ == "__main__":
    main()

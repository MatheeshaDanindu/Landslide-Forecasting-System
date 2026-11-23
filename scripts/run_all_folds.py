"""Driver: train + evaluate fold/model combinations in-process (no subprocess
overhead), appending real results to fold_metrics.csv as each finishes. One
failure (e.g. a fold with no usable patches) is logged and skipped, not fatal
to the rest -- so a partial run still yields real, reportable numbers.

Run several of these as separate OS processes with --shard/--num-shards to
use multiple CPU cores in parallel -- this changes nothing about what gets
trained (same data, same hyperparameters, same code path per fold/model), it
only changes which process trains which fold, so there's no accuracy
tradeoff. Pass --threads to cap each shard's PyTorch intra-op threads so
N parallel shards don't oversubscribe the machine's cores.

    python scripts/run_all_folds.py --shard 0 --num-shards 4 --threads 3
    python scripts/run_all_folds.py --shard 1 --num-shards 4 --threads 3
    ... (one per shard, launched concurrently)
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

import landslide_pipeline.evaluate as evaluate_mod
import landslide_pipeline.train as train_mod

FOLDS = list(range(8))
MODELS = ["baseline", "comparison"]


def run_one(fold: int, model_key: str):
    t0 = time.time()
    print(f"\n=== TRAIN fold={fold} model={model_key} started {time.strftime('%H:%M:%S')} ===", flush=True)
    sys.argv = ["train.py", "--fold", str(fold), "--model-key", model_key]
    train_mod.main()
    print(f"=== TRAIN fold={fold} model={model_key} done in {time.time()-t0:.0f}s ===", flush=True)

    print(f"--- EVAL fold={fold} model={model_key} ---", flush=True)
    sys.argv = ["evaluate.py", "--fold", str(fold), "--model-key", model_key]
    evaluate_mod.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None,
                     help="cap torch intra-op threads; set to cores/num_shards "
                          "when running shards concurrently")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    combos = [(f, m) for f in FOLDS for m in MODELS]
    my_combos = combos[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {my_combos}", flush=True)

    for fold, model_key in my_combos:
        try:
            run_one(fold, model_key)
        except Exception:
            print(f"!!! fold={fold} model={model_key} FAILED, skipping !!!", flush=True)
            traceback.print_exc()
    print(f"\n=== SHARD {args.shard} DONE ===", flush=True)


if __name__ == "__main__":
    main()

"""Driver: train + evaluate all 8 folds x 2 models sequentially, in-process
(no subprocess overhead), appending real results to fold_metrics.csv as each
finishes. One failure (e.g. a fold with no usable patches) is logged and
skipped, not fatal to the rest -- so a partial run still yields real,
reportable numbers instead of nothing.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    for fold in FOLDS:
        for model_key in MODELS:
            try:
                run_one(fold, model_key)
            except Exception:
                print(f"!!! fold={fold} model={model_key} FAILED, skipping !!!", flush=True)
                traceback.print_exc()
    print("\n=== ALL FOLDS DONE ===", flush=True)


if __name__ == "__main__":
    main()

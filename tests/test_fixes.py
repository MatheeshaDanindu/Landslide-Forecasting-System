"""One runnable check per fixed-bug: run directly (`python tests/test_fixes.py`)
or under pytest. No fixtures/framework beyond what's already a dependency."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from shapely.geometry import Polygon

from landslide_pipeline.config import load_project_config
from landslide_pipeline.labels import _oriented_minor_axis
from landslide_pipeline.metrics import segmentation_metrics
from landslide_pipeline.training import dice_loss, focal_loss


def test_minor_axis_is_short_side_not_half_perimeter():
    # a 10 x 100 axis-aligned rectangle: minor axis must be 10, not (10+100)=110
    rect = Polygon([(0, 0), (100, 0), (100, 10), (0, 10)])
    axis = _oriented_minor_axis(rect)
    assert abs(axis - 10.0) < 1e-6, f"expected minor axis 10.0, got {axis}"


def test_config_merge_does_not_collide_on_shared_keys():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_project_config(repo_root)
    # acquisition.yaml and preprocessing.yaml both define a top-level "labels" key --
    # the old flat merge silently dropped one. Namespacing must keep both.
    assert "path" in cfg["acquisition"]["labels"]
    assert "min_minor_axis_m" in cfg["preprocessing"]["labels"]


def test_masked_loss_ignores_masked_pixels():
    pred = torch.zeros(1, 4)
    target = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # all positive
    mask_all_valid = torch.ones(1, 4)
    mask_half_valid = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss_all = focal_loss(pred, target, valid_mask=mask_all_valid)
    loss_half = focal_loss(pred, target, valid_mask=mask_half_valid)
    # pred is identical (all zero logits) and target is identical (all 1) in both
    # halves, so masking out half the (identical) pixels must not change the mean loss
    assert abs(loss_all.item() - loss_half.item()) < 1e-5

    dloss_all = dice_loss(torch.sigmoid(pred).logit(), target, valid_mask=mask_all_valid)
    dloss_half = dice_loss(torch.sigmoid(pred).logit(), target, valid_mask=mask_half_valid)
    assert abs(dloss_all.item() - dloss_half.item()) < 1e-5


def test_segmentation_metrics_excludes_invalid_pixels():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0], dtype=np.float32)  # 1 TP, 1 FN, 1 FP, 1 TN unmasked
    valid = np.array([True, True, True, False])  # drop the last TN pixel
    m = segmentation_metrics(y_true, y_pred, valid_mask=valid)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 0


if __name__ == "__main__":
    test_minor_axis_is_short_side_not_half_perimeter()
    test_config_merge_does_not_collide_on_shared_keys()
    test_masked_loss_ignores_masked_pixels()
    test_segmentation_metrics_excludes_invalid_pixels()
    print("all fix checks passed")

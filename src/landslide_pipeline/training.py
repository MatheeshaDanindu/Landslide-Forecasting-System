from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _masked_mean(loss: torch.Tensor, valid_mask: Optional[torch.Tensor]):
    """Sum(mask*loss)/sum(mask), never dividing by zero (all-invalid patches
    contribute 0, not NaN -- caller should still skip them upstream if possible)."""
    if valid_mask is None:
        return loss.mean()
    mask = valid_mask.view(-1).float()
    denom = mask.sum().clamp(min=1e-8)
    return (mask * loss).sum() / denom


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
               reduction: str = "mean", valid_mask: Optional[torch.Tensor] = None):
    pred = pred.view(-1)
    target = target.view(-1)
    eps = 1e-8
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_factor = alpha * target + (1 - alpha) * (1 - target)
    modulating = (1.0 - p_t).clamp(min=eps, max=1.0) ** gamma
    loss = alpha_factor * modulating * ce
    if reduction == "mean":
        return _masked_mean(loss, valid_mask)
    if reduction == "sum":
        return (loss * valid_mask.view(-1).float()).sum() if valid_mask is not None else loss.sum()
    return loss


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
              valid_mask: Optional[torch.Tensor] = None):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1).float()
    if valid_mask is not None:
        valid = valid_mask.view(-1).float()
        pred, target = pred * valid, target * valid
    intersection = (pred * target).sum()
    cardinality = (pred + target).sum()
    return 1.0 - ((2.0 * intersection + eps) / (cardinality + eps))


def focal_dice_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
                     dice_weight: float = 0.5, valid_mask: Optional[torch.Tensor] = None):
    """valid_mask: optional bool/float tensor, same shape as target, True/1 where the
    pixel is a real (non-cloud/shadow/nodata) observation. Excluded pixels contribute
    zero gradient instead of being silently treated as background negatives."""
    focal = focal_loss(pred, target, alpha=alpha, gamma=gamma, reduction="mean", valid_mask=valid_mask)
    dice = dice_loss(pred, target, valid_mask=valid_mask)
    return (1.0 - dice_weight) * focal + dice_weight * dice


def build_model(cfg: Dict[str, object]):
    import segmentation_models_pytorch as smp

    model_cfg = cfg.get("baseline", {})
    arch = model_cfg.get("arch", "Unet")
    encoder = model_cfg.get("encoder", "resnet34")
    in_channels = int(cfg.get("input_channels", 16))
    arch_cls = getattr(smp, arch)
    return arch_cls(
        encoder_name=encoder,
        encoder_weights=model_cfg.get("encoder_weights"),
        in_channels=in_channels,
        classes=1,
    )


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int, base_seed: int = 42):
    """Pass as DataLoader(..., worker_init_fn=partial(worker_init_fn, base_seed=seed))
    so forked workers don't all replay the same numpy/random stream."""
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
    pred = pred.view(-1)
    target = target.view(-1)
    eps = 1e-8
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_factor = alpha * target + (1 - alpha) * (1 - target)
    modulating = (1.0 - p_t).clamp(min=eps, max=1.0) ** gamma
    loss = alpha_factor * modulating * ce
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred = torch.sigmoid(pred)
    target = target.float()
    intersection = (pred * target).sum()
    cardinality = (pred + target).sum()
    return 1.0 - ((2.0 * intersection + eps) / (cardinality + eps))


def focal_dice_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, dice_weight: float = 0.5):
    focal = focal_loss(pred, target, alpha=alpha, gamma=gamma, reduction="mean")
    dice = dice_loss(pred, target)
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""Model factory. `training.build_model` already does this for a single spec;
this just resolves which spec (`baseline` / `comparison`) by key -- no new
logic, per Issue 1's request for a models.py entry point."""
from __future__ import annotations

from typing import Dict


def build_model_for_key(cfg: Dict, model_key: str):
    import segmentation_models_pytorch as smp
    spec = cfg["model"][model_key]
    arch_cls = getattr(smp, spec["arch"])
    return arch_cls(
        encoder_name=spec["encoder"],
        encoder_weights=spec.get("encoder_weights"),
        in_channels=int(cfg["model"]["input_channels"]),
        classes=1,
    )

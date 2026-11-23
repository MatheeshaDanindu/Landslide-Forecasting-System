from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project_config(repo_root: str | Path | None = None) -> Dict[str, Any]:
    """Namespaced merge, one key per config file.

    A flat `merged.update(data)` silently drops data: both acquisition.yaml
    and preprocessing.yaml define a top-level `labels` key, so the second
    load clobbers the first (losing `labels.path`) with no error -- it only
    "worked" before because callers' hardcoded fallback happened to match
    the real path. Namespacing makes collisions structurally impossible.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    config_dir = root / "configs"
    out: Dict[str, Any] = {}
    for name in ["acquisition.yaml", "preprocessing.yaml", "model.yaml", "spatial_cv.yaml"]:
        key = name.removesuffix(".yaml")
        out[key] = load_yaml(config_dir / name)
    return out

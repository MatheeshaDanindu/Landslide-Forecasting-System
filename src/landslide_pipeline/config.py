from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project_config(repo_root: str | Path | None = None) -> Dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    config_dir = root / "configs"
    merged: Dict[str, Any] = {}
    for name in ["acquisition.yaml", "preprocessing.yaml", "model.yaml", "spatial_cv.yaml"]:
        data = load_yaml(config_dir / name)
        if not data:
            continue
        merged.update(data)
    return merged

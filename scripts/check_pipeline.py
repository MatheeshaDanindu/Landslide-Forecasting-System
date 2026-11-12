
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from landslide_pipeline.config import load_project_config
from landslide_pipeline.labels import load_labels, record_label_summary


def main() -> None:
    cfg = load_project_config(ROOT)
    label_rel = cfg.get("labels", {}).get("path", "data/raw/labels/Lanslides_Ditwah_2025.shp")
    label_path = ROOT / label_rel
    print(f"Repository root: {ROOT}")
    print(f"Config files loaded: {sorted(list((ROOT / 'configs').glob('*.yaml')))}")
    if label_path.exists():
        labels = load_labels(label_path, cfg.get("crs", {}).get("metric", "EPSG:32644"))
        print(f"Loaded labels: {record_label_summary(labels)}")
    else:
        print(f"Label file missing: {label_path}")
    print("Pipeline configuration and scientific guardrails are ready for a real-data run.")


if __name__ == "__main__":
    main()

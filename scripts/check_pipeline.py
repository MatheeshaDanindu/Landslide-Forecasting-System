
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
    label_rel = cfg["acquisition"]["labels"]["path"]
    # labels.path is DATA_DIR-relative (DATA_DIR = ROOT/"data"), not ROOT-relative --
    # see docs/architecture.md; this only ever worked before because the pre-fix
    # config merge bug silently fell back to a hardcoded default that happened to
    # include "data/".
    label_path = ROOT / "data" / label_rel
    print(f"Repository root: {ROOT}")
    print(f"Config files loaded: {sorted(list((ROOT / 'configs').glob('*.yaml')))}")
    if label_path.exists():
        labels = load_labels(label_path, cfg["preprocessing"]["crs"]["metric"])
        min_minor_axis_m = cfg["preprocessing"]["labels"]["min_minor_axis_m"]
        summary = record_label_summary(labels)
        summary["small_objects"] = int((labels["minor_axis_m"] < min_minor_axis_m).sum())
        summary["small_share"] = float((labels["minor_axis_m"] < min_minor_axis_m).mean())
        print(f"Loaded labels: {summary}")
    else:
        print(f"Label file missing: {label_path}")
    print("Pipeline configuration and scientific guardrails are ready for a real-data run.")


if __name__ == "__main__":
    main()

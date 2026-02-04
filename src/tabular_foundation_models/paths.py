"""Central path definitions for the repo. Use these so all scripts refer to the same locations."""

from pathlib import Path

# repo root (src/tabular_foundation_models/paths.py -> parent.parent.parent)
PATH_REPO = Path(__file__).resolve().parent.parent.parent
PATH_DATA = PATH_REPO / "data"
PATH_DATA_RAW = PATH_DATA / "raw"
PATH_DATA_PROCESSED = PATH_DATA / "processed"

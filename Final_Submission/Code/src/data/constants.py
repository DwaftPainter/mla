"""Canonical NUCLE paths and ordered taxonomy (independent of model config)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data_set"
SGML_FILE = DATA_ROOT / "raw" / "nucle3.2.sgml"
ANNOTATIONS_FILE = DATA_ROOT / "interim" / "nucle_errors.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
TARGET_LABELS = ("ArtOrDet", "Prep", "Nn", "Vform", "SVA")
# A filename is not release metadata; supply a verified release explicitly.
DEFAULT_RELEASE = "unverified"

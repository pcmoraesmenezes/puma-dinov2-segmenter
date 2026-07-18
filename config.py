import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

BASE_PATH = Path(os.getenv("PUMA_BASE_PATH", REPO_ROOT / "puma"))
CHECKPOINT_DIR = Path(os.getenv("PUMA_CHECKPOINT_DIR", REPO_ROOT / "checkpoints"))
VIS_DIR = Path(os.getenv("PUMA_VIS_DIR", REPO_ROOT / "visualizations"))

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_linear_probe.pt"
OUTPUT_DIR = VIS_DIR / "predictions"

ROIS_ZIP = BASE_PATH / "ROIs.zip"
NUCLEI_ZIP = BASE_PATH / "nuclei.geojson.zip"
TISSUE_ZIP = BASE_PATH / "tissue.geojson.zip"
FEATURES_DIR = BASE_PATH / "features"
FEATURES_RESNET_DIR = BASE_PATH / "features_resnet50"
MASK_TISSUE_DIR = BASE_PATH / "masks" / "tissue"
MASK_NUCLEI_DIR = BASE_PATH / "masks" / "nuclei"

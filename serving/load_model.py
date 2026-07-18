"""Reusable loader for the PUMA DinoV2 segmentation pipeline.

Bundles the frozen DINOv2 backbone (feature extractor) with the trained
PUMA segmentation head, and validates the pair with a dummy forward pass.
Intended to be imported by the future serving layer (FastAPI) as well as
run standalone for a quick health check.
"""
import os
import sys
import time
import importlib

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "pipeline"))

from utils.logger import get_logger
from config import CHECKPOINT_PATH

logger = get_logger(__name__)

model_module = importlib.import_module("04_model")
DinoPUMASegmenter = model_module.DinoPUMASegmenter

BACKBONE_NAME = "dinov2_vits14"
IN_CHANNELS = 384
TISSUE_CLASSES = 6
NUCLEI_CLASSES = 4
BACKBONE_INPUT_SIZE = 1036  # multiple of 14 (patch size), see pipeline/02_feature_extractor.py
HEAD_OUTPUT_SIZE = 1024  # training mask resolution, see pipeline/03_dataset.py


class ModelBundle:
    """Bundles the frozen backbone (DINOv2) and the trained head (DinoPUMASegmenter)."""

    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module, device: torch.device):
        self.backbone = backbone
        self.head = head
        self.device = device

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """image_tensor: (B, 3, 1036, 1036) preprocessed -> (B, 384, 74, 74)."""
        with torch.no_grad():
            out = self.backbone.get_intermediate_layers(image_tensor, n=1)[0]
            b, l, c = out.shape
            patch_h = patch_w = int(l ** 0.5)
            return out.reshape(b, patch_h, patch_w, c).permute(0, 3, 1, 2)

    def predict(self, image_tensor: torch.Tensor, upscale: bool = True) -> dict:
        features = self.extract_features(image_tensor)
        with torch.no_grad():
            return self.head(features, upscale=upscale)


def load_backbone(device: torch.device) -> torch.nn.Module:
    logger.info(f"Loading DINOv2 backbone ({BACKBONE_NAME})...")
    backbone = torch.hub.load("facebookresearch/dinov2", BACKBONE_NAME)
    backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def load_head(device: torch.device) -> tuple:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    head = DinoPUMASegmenter(
        in_channels=IN_CHANNELS,
        tissue_classes=TISSUE_CLASSES,
        nuclei_classes=NUCLEI_CLASSES,
    ).to(device)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()

    logger.info(
        f"Head loaded — epoch {checkpoint.get('epoch', '?')}, "
        f"val loss {checkpoint.get('loss', float('nan')):.4f}"
    )
    return head, checkpoint


def healthcheck(bundle: ModelBundle) -> dict:
    """Runs a dummy forward pass (backbone + head) to validate the loaded pair is coherent."""
    start = time.time()
    dummy_image = torch.randn(1, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE, device=bundle.device)

    outputs = bundle.predict(dummy_image, upscale=True)
    elapsed = time.time() - start

    expected_tissue_shape = (1, TISSUE_CLASSES, HEAD_OUTPUT_SIZE, HEAD_OUTPUT_SIZE)
    expected_nuclei_shape = (1, NUCLEI_CLASSES, HEAD_OUTPUT_SIZE, HEAD_OUTPUT_SIZE)

    tissue_shape = tuple(outputs["tissue"].shape)
    nuclei_shape = tuple(outputs["nuclei"].shape)
    ok = tissue_shape == expected_tissue_shape and nuclei_shape == expected_nuclei_shape

    result = {
        "ok": ok,
        "elapsed_seconds": round(elapsed, 3),
        "tissue_shape": tissue_shape,
        "nuclei_shape": nuclei_shape,
    }

    if ok:
        logger.info(f"Healthcheck OK: {result}")
    else:
        logger.error(f"Healthcheck FAILED (shape mismatch): {result}")

    return result


def load_model(device: torch.device = None) -> ModelBundle:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    backbone = load_backbone(device)
    head, _ = load_head(device)

    bundle = ModelBundle(backbone=backbone, head=head, device=device)
    result = healthcheck(bundle)

    if not result["ok"]:
        raise RuntimeError(f"Model healthcheck failed: {result}")

    return bundle


if __name__ == "__main__":
    load_model()
    logger.info("load_model.py: pipeline ready to serve.")

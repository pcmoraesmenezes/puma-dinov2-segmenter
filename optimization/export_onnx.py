"""Exports the backbone (DINOv2) and head (DinoPUMASegmenter) to ONNX.

Generates checkpoints/backbone.onnx and checkpoints/head.onnx, used by
eval_onnx.py to compare latency against the PyTorch eager baseline.
"""
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_model import load_backbone, load_head, BACKBONE_INPUT_SIZE, IN_CHANNELS
from config import CHECKPOINT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

BACKBONE_ONNX_PATH = CHECKPOINT_DIR / "backbone.onnx"
HEAD_ONNX_PATH = CHECKPOINT_DIR / "head.onnx"


class BackboneWrapper(torch.nn.Module):
    """Wraps get_intermediate_layers (a custom method, not the default forward) into an exportable forward."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.get_intermediate_layers(x, n=1)[0]
        b, l, c = out.shape
        patch_h = patch_w = int(l**0.5)
        return out.reshape(b, patch_h, patch_w, c).permute(0, 3, 1, 2)


class HeadWrapper(torch.nn.Module):
    """upscale=True fixed — ONNX requires a static graph, and this is the real serving path."""

    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = head

    def forward(self, features: torch.Tensor):
        out = self.head(features, upscale=True)
        return out["tissue"], out["nuclei"]


def main():
    device = torch.device("cpu")

    logger.info("Loading backbone + head (FP32)...")
    backbone = load_backbone(device)
    head, _ = load_head(device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    logger.info(f"Exporting backbone to {BACKBONE_ONNX_PATH}...")
    dummy_image = torch.randn(1, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE)
    torch.onnx.export(
        BackboneWrapper(backbone),
        dummy_image,
        str(BACKBONE_ONNX_PATH),
        input_names=["image"],
        output_names=["features"],
        opset_version=17,
    )
    logger.info("Backbone exported.")

    logger.info(f"Exporting head to {HEAD_ONNX_PATH}...")
    dummy_features = torch.randn(1, IN_CHANNELS, 74, 74)
    torch.onnx.export(
        HeadWrapper(head),
        dummy_features,
        str(HEAD_ONNX_PATH),
        input_names=["features"],
        output_names=["tissue", "nuclei"],
        opset_version=17,
    )
    logger.info("Head exported.")


if __name__ == "__main__":
    main()

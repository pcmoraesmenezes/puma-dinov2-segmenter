"""Dynamic quantization (int8) measured on the real harness — latency AND F1, not just dummy input."""
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_model import load_backbone, load_head, ModelBundle
from eval_utils import evaluate_method, get_val_ids
from utils.logger import get_logger

logger = get_logger(__name__)


def make_bundle(device):
    backbone = load_backbone(device)
    head, _ = load_head(device)
    backbone_int8 = torch.quantization.quantize_dynamic(backbone, {torch.nn.Linear}, dtype=torch.qint8)
    return ModelBundle(backbone_int8, head, device)


def run():
    device = torch.device("cpu")
    bundle = make_bundle(device)
    val_ids = get_val_ids()

    def predict_fn(tensor):
        return bundle.predict(tensor, upscale=True)

    return evaluate_method(predict_fn, "Dynamic Quantization (INT8)", val_ids, resolution=1036)


if __name__ == "__main__":
    run()

"""ONNX Runtime (CPU) measured on the real harness — latency AND F1."""
import os
import sys

import torch
import onnxruntime as ort

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from export_onnx import BACKBONE_ONNX_PATH, HEAD_ONNX_PATH
from eval_utils import evaluate_method, get_val_ids
from utils.logger import get_logger

logger = get_logger(__name__)


def run():
    if not BACKBONE_ONNX_PATH.exists() or not HEAD_ONNX_PATH.exists():
        raise FileNotFoundError("Run optimization/export_onnx.py first.")

    so = ort.SessionOptions()
    backbone_sess = ort.InferenceSession(str(BACKBONE_ONNX_PATH), so, providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(str(HEAD_ONNX_PATH), so, providers=["CPUExecutionProvider"])

    val_ids = get_val_ids()

    def predict_fn(tensor):
        image_np = tensor.numpy()
        features = backbone_sess.run(None, {"image": image_np})[0]
        tissue, nuclei = head_sess.run(None, {"features": features})
        return {"tissue": torch.from_numpy(tissue), "nuclei": torch.from_numpy(nuclei)}

    return evaluate_method(predict_fn, "ONNX Runtime (CPU)", val_ids, resolution=1036)


if __name__ == "__main__":
    run()

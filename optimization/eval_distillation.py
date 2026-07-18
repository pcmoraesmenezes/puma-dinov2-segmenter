"""Evaluates the distilled student (small backbone) + original trained head on the real harness."""
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_model import load_head
from student_model import StudentBackbone, STUDENT_CHECKPOINT_PATH
from eval_utils import evaluate_method, get_val_ids
from utils.logger import get_logger

logger = get_logger(__name__)


def run():
    device = torch.device("cpu")

    if not STUDENT_CHECKPOINT_PATH.exists():
        raise FileNotFoundError("Run optimization/distillation.py first.")

    checkpoint = torch.load(STUDENT_CHECKPOINT_PATH, map_location=device)
    student = StudentBackbone().to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    logger.info(f"Student loaded — epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")

    head, _ = load_head(device)
    val_ids = get_val_ids()

    def predict_fn(tensor):
        with torch.no_grad():
            features = student(tensor)
            return head(features, upscale=True)

    return evaluate_method(predict_fn, "Knowledge Distillation (student CNN)", val_ids, resolution=1036)


if __name__ == "__main__":
    run()

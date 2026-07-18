"""Shared evaluation utilities — used by every model compression benchmark
(quantization, ONNX, hybrid, low-rank, pruning, distillation).

Ensures every method is measured under the SAME protocol: same 41 real
validation images, same ground truth masks, same metric (macro F1).
"""
import io
import os
import sys
import time
import zipfile
import importlib

import cv2
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "pipeline"))

from config import BASE_PATH, ROIS_ZIP, MASK_TISSUE_DIR, MASK_NUCLEI_DIR
from preprocessing import make_transform
from utils.logger import get_logger

logger = get_logger(__name__)

dataset_module = importlib.import_module("03_dataset")
PUMAFeatureDataset = dataset_module.PUMAFeatureDataset

train_module = importlib.import_module("05_train")
calculate_pixel_metrics = train_module.calculate_pixel_metrics


def get_val_ids():
    full_dataset = PUMAFeatureDataset(BASE_PATH, cache_in_ram=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    return [full_dataset.sample_ids[i] for i in val_dataset.indices]


def load_raw_image(image_id):
    with zipfile.ZipFile(ROIS_ZIP, "r") as z:
        candidates = [f for f in z.namelist() if image_id in f and f.endswith(".tif")]
        with z.open(candidates[0]) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")


def load_ground_truth(image_id):
    tissue = cv2.imread(str(MASK_TISSUE_DIR / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
    nuclei = cv2.imread(str(MASK_NUCLEI_DIR / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
    return torch.from_numpy(tissue).long().unsqueeze(0), torch.from_numpy(nuclei).long().unsqueeze(0)


def evaluate_method(predict_fn, method_name: str, val_ids=None, resolution: int = 1036):
    """predict_fn(tensor) -> dict with 'tissue' and 'nuclei' (logits, upscaled to 1024x1024).

    Measures latency (predict_fn only) and real F1 against ground truth, on the
    same validation images used by every other method.
    """
    val_ids = val_ids or get_val_ids()
    transform = make_transform(resolution)

    tissue_accs, tissue_f1s, nuclei_accs, nuclei_f1s, times = [], [], [], [], []

    for image_id in val_ids:
        img = load_raw_image(image_id)
        tensor = transform(img).unsqueeze(0)

        start = time.time()
        outputs = predict_fn(tensor)
        times.append(time.time() - start)

        tissue_gt, nuclei_gt = load_ground_truth(image_id)
        t_acc, t_f1 = calculate_pixel_metrics(outputs["tissue"], tissue_gt, num_classes=6)
        n_acc, n_f1 = calculate_pixel_metrics(outputs["nuclei"], nuclei_gt, num_classes=4)

        tissue_accs.append(t_acc)
        tissue_f1s.append(t_f1)
        nuclei_accs.append(n_acc)
        nuclei_f1s.append(n_f1)

    result = {
        "method": method_name,
        "avg_time": sum(times) / len(times),
        "tissue_acc": sum(tissue_accs) / len(tissue_accs),
        "tissue_f1": sum(tissue_f1s) / len(tissue_f1s),
        "nuclei_acc": sum(nuclei_accs) / len(nuclei_accs),
        "nuclei_f1": sum(nuclei_f1s) / len(nuclei_f1s),
    }
    logger.info(
        f"[{method_name}] latency={result['avg_time']:.3f}s | "
        f"tissue_f1={result['tissue_f1']:.4f} | nuclei_f1={result['nuclei_f1']:.4f}"
    )
    return result

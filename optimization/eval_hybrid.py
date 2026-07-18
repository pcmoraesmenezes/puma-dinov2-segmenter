"""Hybrid strategy: tissue at low resolution (near-zero loss) + nuclei at
intermediate resolution (preserves more F1 than 266, faster than 1036).

Two backbone forward passes per image, each feeding only the relevant head.
"""
import os
import sys

import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_model import load_backbone, load_head
from eval_utils import get_val_ids, make_transform, load_raw_image, load_ground_truth, calculate_pixel_metrics
from utils.logger import get_logger

logger = get_logger(__name__)

TISSUE_RESOLUTION = 266
NUCLEI_RESOLUTION = 728


def extract_features(backbone, image_tensor):
    with torch.no_grad():
        out = backbone.get_intermediate_layers(image_tensor, n=1)[0]
        b, l, c = out.shape
        ph = pw = int(l**0.5)
        return out.reshape(b, ph, pw, c).permute(0, 3, 1, 2)


def run():
    device = torch.device("cpu")
    backbone = load_backbone(device)
    head, _ = load_head(device)

    tissue_transform = make_transform(TISSUE_RESOLUTION)
    nuclei_transform = make_transform(NUCLEI_RESOLUTION)

    val_ids = get_val_ids()

    # eval_utils.evaluate_method assumes 1 fixed resolution; here we need 2 transforms
    # per image (one per head), so the loop is local, reusing eval_utils' building blocks.
    tissue_accs, tissue_f1s, nuclei_accs, nuclei_f1s, times = [], [], [], [], []

    for image_id in val_ids:
        img = load_raw_image(image_id)
        tissue_tensor = tissue_transform(img).unsqueeze(0)
        nuclei_tensor = nuclei_transform(img).unsqueeze(0)

        start = time.time()
        with torch.no_grad():
            tissue_features = extract_features(backbone, tissue_tensor)
            tissue_out = head.tissue_head(tissue_features, upscale=True)

            nuclei_features = extract_features(backbone, nuclei_tensor)
            nuclei_out = head.nuclei_head(nuclei_features, upscale=True)
        times.append(time.time() - start)

        tissue_gt, nuclei_gt = load_ground_truth(image_id)
        t_acc, t_f1 = calculate_pixel_metrics(tissue_out, tissue_gt, num_classes=6)
        n_acc, n_f1 = calculate_pixel_metrics(nuclei_out, nuclei_gt, num_classes=4)

        tissue_accs.append(t_acc)
        tissue_f1s.append(t_f1)
        nuclei_accs.append(n_acc)
        nuclei_f1s.append(n_f1)

    result = {
        "method": f"Hybrid (tissue@{TISSUE_RESOLUTION}, nuclei@{NUCLEI_RESOLUTION})",
        "avg_time": sum(times) / len(times),
        "tissue_acc": sum(tissue_accs) / len(tissue_accs),
        "tissue_f1": sum(tissue_f1s) / len(tissue_f1s),
        "nuclei_acc": sum(nuclei_accs) / len(nuclei_accs),
        "nuclei_f1": sum(nuclei_f1s) / len(nuclei_f1s),
    }
    logger.info(
        f"[{result['method']}] latency={result['avg_time']:.3f}s | "
        f"tissue_f1={result['tissue_f1']:.4f} | nuclei_f1={result['nuclei_f1']:.4f}"
    )
    return result


if __name__ == "__main__":
    run()

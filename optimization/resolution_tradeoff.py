"""Latency vs. accuracy trade-off across input resolutions.

Measures, on the REAL validation dataset (not dummy tensors), how much
F1/accuracy is lost when reducing the backbone's input resolution — to decide
whether the latency optimization is viable without sacrificing the clinical
task (nuclei, 15px margin). Generates and saves the trade-off figure in visualizations/.
"""
import io
import os
import sys
import time
import zipfile
import importlib

import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torchvision.transforms as T
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "pipeline"))

from load_model import load_backbone, load_head, ModelBundle
from config import BASE_PATH, ROIS_ZIP, MASK_TISSUE_DIR, MASK_NUCLEI_DIR, VIS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

dataset_module = importlib.import_module("03_dataset")
PUMAFeatureDataset = dataset_module.PUMAFeatureDataset

train_module = importlib.import_module("05_train")
calculate_pixel_metrics = train_module.calculate_pixel_metrics

RESOLUTIONS = [1036, 728, 518, 266]
DOD_LATENCY_SECONDS = 1.0
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Validated palette (dataviz skill) — slot 1 blue, slot 2 green
COLOR_LATENCY = "#2a78d6"
COLOR_TISSUE = "#2a78d6"
COLOR_NUCLEI = "#008300"
COLOR_MUTED = "#898781"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e1e0d9"


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


def evaluate_resolution(bundle, size, val_ids):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    tissue_accs, tissue_f1s, nuclei_accs, nuclei_f1s, times = [], [], [], [], []

    for image_id in val_ids:
        img = load_raw_image(image_id)
        tensor = transform(img).unsqueeze(0)

        start = time.time()
        outputs = bundle.predict(tensor, upscale=True)
        times.append(time.time() - start)

        tissue_gt, nuclei_gt = load_ground_truth(image_id)
        t_acc, t_f1 = calculate_pixel_metrics(outputs["tissue"], tissue_gt, num_classes=6)
        n_acc, n_f1 = calculate_pixel_metrics(outputs["nuclei"], nuclei_gt, num_classes=4)

        tissue_accs.append(t_acc)
        tissue_f1s.append(t_f1)
        nuclei_accs.append(n_acc)
        nuclei_f1s.append(n_f1)

    result = {
        "resolution": size,
        "avg_time": sum(times) / len(times),
        "tissue_acc": sum(tissue_accs) / len(tissue_accs),
        "tissue_f1": sum(tissue_f1s) / len(tissue_f1s),
        "nuclei_acc": sum(nuclei_accs) / len(nuclei_accs),
        "nuclei_f1": sum(nuclei_f1s) / len(nuclei_f1s),
    }
    logger.info(
        f"[{size}x{size}] latency={result['avg_time']:.3f}s | "
        f"tissue_f1={result['tissue_f1']:.4f} | nuclei_f1={result['nuclei_f1']:.4f}"
    )
    return result


def plot_tradeoff(results, save_path):
    resolutions = [r["resolution"] for r in results]
    latencies = [r["avg_time"] for r in results]
    tissue_f1s = [r["tissue_f1"] for r in results]
    nuclei_f1s = [r["nuclei_f1"] for r in results]

    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax_latency, ax_f1) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Input Resolution Trade-off: Latency vs. Segmentation Quality",
        fontsize=14, fontweight="bold", color="#0b0b0b",
    )

    # --- Panel A: Latency ---
    ax_latency.plot(
        resolutions, latencies, color=COLOR_LATENCY, linewidth=2,
        marker="o", markersize=8, markerfacecolor=COLOR_LATENCY,
        markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=3,
    )
    ax_latency.axhline(
        DOD_LATENCY_SECONDS, color=COLOR_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1,
    )
    ax_latency.text(
        min(resolutions), DOD_LATENCY_SECONDS, "  1s target (DoD)",
        color=COLOR_TEXT_SECONDARY, fontsize=9, va="bottom", ha="left",
    )
    for x, y in zip(resolutions, latencies):
        ax_latency.annotate(
            f"{y:.2f}s", (x, y), textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9, color=COLOR_TEXT_SECONDARY,
        )
    ax_latency.set_title("Backbone + Head Latency", fontsize=11, color="#0b0b0b")
    ax_latency.set_xlabel("Input resolution (px)", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax_latency.set_ylabel("Latency (seconds)", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax_latency.set_xticks(resolutions)
    ax_latency.set_xticklabels([f"{r}×{r}" for r in resolutions])
    ax_latency.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax_latency.spines[["top", "right"]].set_visible(False)
    ax_latency.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax_latency.tick_params(colors=COLOR_MUTED)

    # --- Panel B: F1 (tissue + nuclei) ---
    ax_f1.plot(
        resolutions, tissue_f1s, color=COLOR_TISSUE, linewidth=2,
        marker="o", markersize=8, markerfacecolor=COLOR_TISSUE,
        markeredgecolor="#fcfcfb", markeredgewidth=2, label="Tissue F1", zorder=3,
    )
    ax_f1.plot(
        resolutions, nuclei_f1s, color=COLOR_NUCLEI, linewidth=2,
        marker="o", markersize=8, markerfacecolor=COLOR_NUCLEI,
        markeredgecolor="#fcfcfb", markeredgewidth=2, label="Nuclei F1", zorder=3,
    )
    ax_f1.set_title("Segmentation Quality (Macro F1)", fontsize=11, color="#0b0b0b")
    ax_f1.set_xlabel("Input resolution (px)", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax_f1.set_ylabel("Macro F1 score", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax_f1.set_xticks(resolutions)
    ax_f1.set_xticklabels([f"{r}×{r}" for r in resolutions])
    ax_f1.set_ylim(0, 1)
    ax_f1.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax_f1.spines[["top", "right"]].set_visible(False)
    ax_f1.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax_f1.tick_params(colors=COLOR_MUTED)
    legend = ax_f1.legend(frameon=False, loc="lower right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_SECONDARY)

    fig.text(
        0.5, -0.02,
        "Backbone: DINOv2 ViT-S/14 (frozen) · Head: DinoPUMASegmenter · CPU-only inference, 41 validation images · 17/07/2026",
        ha="center", fontsize=8, color=COLOR_MUTED,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    logger.info(f"Figure saved: {save_path}")


def main():
    device = torch.device("cpu")
    backbone = load_backbone(device)
    head, _ = load_head(device)
    bundle = ModelBundle(backbone, head, device)

    val_ids = get_val_ids()
    logger.info(f"Validation set: {len(val_ids)} images. Testing resolutions: {RESOLUTIONS}")

    results = [evaluate_resolution(bundle, size, val_ids) for size in RESOLUTIONS]

    os.makedirs(VIS_DIR, exist_ok=True)
    save_path = VIS_DIR / "resolution_latency_accuracy_tradeoff.png"
    plot_tradeoff(results, save_path)

    logger.info("=" * 80)
    logger.info(f"{'Resolution':^12} | {'Latency':^10} | {'Tissue F1':^10} | {'Nuclei F1':^10} | {'DoD (<1s)':^10}")
    logger.info("=" * 80)
    for r in results:
        dod = "OK" if r["avg_time"] < DOD_LATENCY_SECONDS else "FAIL"
        logger.info(
            f"{r['resolution']:^12} | {r['avg_time']:^10.3f} | {r['tissue_f1']:^10.4f} | "
            f"{r['nuclei_f1']:^10.4f} | {dod:^10}"
        )


if __name__ == "__main__":
    main()

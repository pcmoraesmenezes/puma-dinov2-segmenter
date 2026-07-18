import sys
import os
import zipfile
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torch.utils.data import random_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
model_module = importlib.import_module("04_model")
DinoPUMASegmenter = model_module.DinoPUMASegmenter
dataset_module = importlib.import_module("03_dataset")
PUMAFeatureDataset = dataset_module.PUMAFeatureDataset

from utils.logger import get_logger
from config import BASE_PATH, CHECKPOINT_PATH, OUTPUT_DIR, FEATURES_DIR, ROIS_ZIP

logger = get_logger(__name__)

NUM_SAMPLES = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palettes for each class
TISSUE_COLORS = {
    0: (0.9, 0.9, 0.9, 0.0),   # background — transparente
    1: (1.0, 0.85, 0.0, 0.5),  # stroma — amarelo
    2: (0.9, 0.1, 0.1, 0.6),   # blood vessel — vermelho
    3: (0.5, 0.0, 0.8, 0.5),   # tumor — roxo
    4: (0.1, 0.8, 0.1, 0.5),   # epidermis — verde
    5: (1.0, 0.5, 0.0, 0.6),   # necrosis — laranja
}

NUCLEI_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),   # background — transparente
    1: (0.9, 0.1, 0.1, 0.7),   # tumor nucleus — vermelho
    2: (0.0, 0.8, 0.9, 0.7),   # lymphocyte/plasma — ciano
    3: (1.0, 0.9, 0.0, 0.7),   # other — amarelo
}

TISSUE_LABELS = {1: "Stroma", 2: "Blood Vessel", 3: "Tumor", 4: "Epidermis", 5: "Necrosis"}
NUCLEI_LABELS = {1: "Tumor Nucleus", 2: "Lymphocyte/Plasma", 3: "Other"}


def mask_to_rgba(mask_np, color_map):
    h, w = mask_np.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for class_id, color in color_map.items():
        region = mask_np == class_id
        rgba[region] = color
    return rgba


def run_inference(model, features_path, device):
    features = torch.load(features_path, map_location=device).squeeze(0).unsqueeze(0)
    with torch.no_grad():
        outputs = model(features, upscale=True)
    tissue_pred = torch.argmax(outputs["tissue"], dim=1).squeeze(0).cpu().numpy()
    nuclei_pred = torch.argmax(outputs["nuclei"], dim=1).squeeze(0).cpu().numpy()
    return tissue_pred, nuclei_pred


def load_original_image(image_id, rois_zip_path):
    with zipfile.ZipFile(rois_zip_path, 'r') as z:
        candidates = [f for f in z.namelist() if image_id in f and f.endswith('.tif')]
        if not candidates:
            return None
        with z.open(candidates[0]) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")


def plot_prediction(image_id, original_img, tissue_pred, nuclei_pred, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f"Prediction: {image_id}", fontsize=14, fontweight="bold")

    # Original
    axes[0].imshow(original_img)
    axes[0].set_title("Original (H&E)", fontsize=11)
    axes[0].axis("off")

    # Tissue overlay
    axes[1].imshow(original_img)
    axes[1].imshow(mask_to_rgba(tissue_pred, TISSUE_COLORS))
    axes[1].set_title("Tissue Segmentation", fontsize=11)
    axes[1].axis("off")
    legend_patches = [mpatches.Patch(color=TISSUE_COLORS[k][:3], label=v) for k, v in TISSUE_LABELS.items()]
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=7)

    # Nuclei overlay
    axes[2].imshow(original_img)
    axes[2].imshow(mask_to_rgba(nuclei_pred, NUCLEI_COLORS))
    axes[2].set_title("Nuclei Segmentation", fontsize=11)
    axes[2].axis("off")
    legend_patches = [mpatches.Patch(color=NUCLEI_COLORS[k][:3], label=v) for k, v in NUCLEI_LABELS.items()]
    axes[2].legend(handles=legend_patches, loc="lower right", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{image_id}_prediction.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = DinoPUMASegmenter(in_channels=384, tissue_classes=6, nuclei_classes=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded from epoch {checkpoint['epoch']} (val loss: {checkpoint['loss']:.4f})")

    # Rebuild the split with the same training seed, to guarantee validation samples
    full_dataset = PUMAFeatureDataset(BASE_PATH, cache_in_ram=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_ids = [full_dataset.sample_ids[i] for i in val_dataset.indices]
    logger.info(f"Validation set: {len(val_ids)} samples. Running on first {NUM_SAMPLES}.")

    for image_id in val_ids[:NUM_SAMPLES]:
        logger.info(f"Running inference: {image_id}")

        features_path = os.path.join(FEATURES_DIR, f"{image_id}.pt")
        tissue_pred, nuclei_pred = run_inference(model, features_path, device)

        original_img = load_original_image(image_id, ROIS_ZIP)
        if original_img is None:
            logger.error(f"Image not found in zip: {image_id}")
            continue

        plot_prediction(image_id, original_img, tissue_pred, nuclei_pred, OUTPUT_DIR)

    logger.info(f"Done. {NUM_SAMPLES} predictions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

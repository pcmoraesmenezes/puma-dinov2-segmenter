import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import BASE_PATH, CHECKPOINT_DIR, VIS_DIR

logger = get_logger(__name__)

import importlib
dataset_module = importlib.import_module("03_dataset")
PUMAFeatureDataset = dataset_module.PUMAFeatureDataset

model_module = importlib.import_module("04_model")
DinoPUMASegmenter = model_module.DinoPUMASegmenter

def calculate_pixel_metrics(preds, targets, num_classes):
    """
    Computes pixel-wise accuracy and macro F1, optimized for PyTorch tensors.
    """
    preds_labels = torch.argmax(preds, dim=1) # -> (B, H, W)

    # Pixel-wise Accuracy
    correct = (preds_labels == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total

    # Pixel-wise Macro F1 (loop over a handful of classes - no sklearn overhead)
    f1_list = []
    for c in range(num_classes):
        tp = ((preds_labels == c) & (targets == c)).sum().item()
        fp = ((preds_labels == c) & (targets != c)).sum().item()
        fn = ((preds_labels != c) & (targets == c)).sum().item()

        # If the class is absent from both predictions and target, skip it
        if tp + fp + fn == 0:
            continue

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0.0
    return accuracy, macro_f1

def train_one_epoch(model, dataloader, optimizer, criterion_tissue, criterion_nuclei, device):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        features = batch["features"].to(device)
        mask_tissue = batch["mask_tissue"].to(device)
        mask_nuclei = batch["mask_nuclei"].to(device)

        # Downsample targets to 74x74 (DinoV2 feature size) to speed up CPU training
        mask_tissue_small = F.interpolate(mask_tissue.unsqueeze(1).float(), size=(74, 74), mode='nearest').squeeze(1).long()
        mask_nuclei_small = F.interpolate(mask_nuclei.unsqueeze(1).float(), size=(74, 74), mode='nearest').squeeze(1).long()

        optimizer.zero_grad()

        # Forward pass (upscale=False saves 190x the processing)
        outputs = model(features, upscale=False)

        # Individual losses at native resolution
        loss_tissue = criterion_tissue(outputs["tissue"], mask_tissue_small)
        loss_nuclei = criterion_nuclei(outputs["nuclei"], mask_nuclei_small)

        # Joint loss
        loss = loss_tissue + loss_nuclei

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion_tissue, criterion_nuclei, device):
    model.eval()
    val_loss = 0.0

    total_tissue_acc = 0.0
    total_tissue_f1 = 0.0
    total_nuclei_acc = 0.0
    total_nuclei_f1 = 0.0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            mask_tissue = batch["mask_tissue"].to(device)
            mask_nuclei = batch["mask_nuclei"].to(device)

            # Downsample targets to 74x74
            mask_tissue_small = F.interpolate(mask_tissue.unsqueeze(1).float(), size=(74, 74), mode='nearest').squeeze(1).long()
            mask_nuclei_small = F.interpolate(mask_nuclei.unsqueeze(1).float(), size=(74, 74), mode='nearest').squeeze(1).long()

            outputs = model(features, upscale=False)

            # Losses at native resolution
            loss_tissue = criterion_tissue(outputs["tissue"], mask_tissue_small)
            loss_nuclei = criterion_nuclei(outputs["nuclei"], mask_nuclei_small)
            loss = loss_tissue + loss_nuclei

            val_loss += loss.item()

            # Tissue metrics (6 classes)
            t_acc, t_f1 = calculate_pixel_metrics(outputs["tissue"], mask_tissue_small, num_classes=6)
            total_tissue_acc += t_acc
            total_tissue_f1 += t_f1

            # Nuclei metrics (4 classes)
            n_acc, n_f1 = calculate_pixel_metrics(outputs["nuclei"], mask_nuclei_small, num_classes=4)
            total_nuclei_acc += n_acc
            total_nuclei_f1 += n_f1

    num_batches = len(dataloader)
    metrics = {
        "loss": val_loss / num_batches,
        "tissue_acc": total_tissue_acc / num_batches,
        "tissue_f1": total_tissue_f1 / num_batches,
        "nuclei_acc": total_nuclei_acc / num_batches,
        "nuclei_f1": total_nuclei_f1 / num_batches
    }

    return metrics

def save_training_plots(history, save_dir, backbone):
    """
    Generates polished charts of the training/validation metrics and saves them as an image.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Matplotlib aesthetic configuration
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training History - {backbone.upper()} Linear Probing", fontsize=16, fontweight='bold', color='#2c3e50')

    # 1. Loss plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2.5, marker='o')
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", color="#3498db", linewidth=2.5, marker='s')
    axes[0].set_title("Loss Evolution (Joint Loss)", fontsize=12, fontweight='semibold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Joint CrossEntropy")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Tissue metrics plot
    axes[1].plot(epochs, history["tissue_acc"], label="Accuracy", color="#2ecc71", linewidth=2, marker='^')
    axes[1].plot(epochs, history["tissue_f1"], label="Macro F1", color="#27ae60", linewidth=2, marker='v')
    axes[1].set_title("Tissue Segmentation (6 classes)", fontsize=12, fontweight='semibold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (0-1)")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Nuclei metrics plot
    axes[2].plot(epochs, history["nuclei_acc"], label="Accuracy", color="#9b59b6", linewidth=2, marker='o')
    axes[2].plot(epochs, history["nuclei_f1"], label="Macro F1", color="#8e44ad", linewidth=2, marker='s')
    axes[2].set_title("Nuclei Segmentation (4 classes)", fontsize=12, fontweight='semibold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score (0-1)")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"training_history_{backbone}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history plot saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="DinoPUMASegmenter training")
    parser.add_argument("--backbone", type=str, choices=["dinov2", "resnet50"], default="dinov2", help="Feature extraction backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()

    logger.info(f"Initializing {args.backbone.upper()} training pipeline...")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Dataset & DataLoaders
    feature_dir_name = "features" if args.backbone == "dinov2" else "features_resnet50"
    dataset = PUMAFeatureDataset(BASE_PATH, feature_dir_name=feature_dir_name, cache_in_ram=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Reproducible split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val.")

    # 2. Model Initialization
    in_channels = 384 if args.backbone == "dinov2" else 1024
    model = DinoPUMASegmenter(in_channels=in_channels, tissue_classes=6, nuclei_classes=4).to(device)

    # 3. Optimizer, Scheduler & Criteria
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Halves the LR after 2 epochs with no improvement in val loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    criterion_tissue = nn.CrossEntropyLoss()
    criterion_nuclei = nn.CrossEntropyLoss()

    # Metrics history for visualization
    history = {
        "train_loss": [],
        "val_loss": [],
        "tissue_acc": [],
        "tissue_f1": [],
        "nuclei_acc": [],
        "nuclei_f1": []
    }

    # Progress table
    logger.info("=" * 85)
    logger.info(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Tissue Acc':^10} | {'Tissue F1':^10} | {'Nuclei Acc':^10} | {'Nuclei F1':^10} | {'Time (s)':^8}")
    logger.info("=" * 85)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_tissue, criterion_nuclei, device)
        val_metrics = validate(model, val_loader, criterion_tissue, criterion_nuclei, device)

        epoch_time = time.time() - start_time

        # Record metrics in the history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["tissue_acc"].append(val_metrics["tissue_acc"])
        history["tissue_f1"].append(val_metrics["tissue_f1"])
        history["nuclei_acc"].append(val_metrics["nuclei_acc"])
        history["nuclei_f1"].append(val_metrics["nuclei_f1"])

        # Formatted epoch print
        logger.info(
            f"{epoch:^6} | {train_loss:^10.4f} | {val_metrics['loss']:^10.4f} | "
            f"{val_metrics['tissue_acc']:^10.4f} | {val_metrics['tissue_f1']:^10.4f} | "
            f"{val_metrics['nuclei_acc']:^10.4f} | {val_metrics['nuclei_f1']:^10.4f} | "
            f"{epoch_time:^8.1f}"
        )

        scheduler.step(val_metrics["loss"])
        logger.debug(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save the best model
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            checkpoint_name = "best_linear_probe.pt" if args.backbone == "dinov2" else "best_resnet50_probe.pt"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)

    logger.info("=" * 85)
    logger.info(f"Training complete. Best checkpoint: {checkpoint_path}")

    # Generate and save the learning curve plots
    save_training_plots(history, VIS_DIR, args.backbone)

if __name__ == "__main__":
    main()

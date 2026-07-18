"""Final comparison: every compression/optimization method tested in the
2026-07-17 session, under the same protocol (41 real validation images, macro F1).

Results measured earlier in this session via optimization/eval_*.py and
optimization/resolution_tradeoff.py — not recomputed here (each run is already
in the logs), just consolidated and plotted.
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# method, latency(s), tissue_f1, nuclei_f1, is_baseline
RESULTS = [
    ("Baseline (1036px, FP32)",              5.274, 0.4898, 0.4013, True),
    ("Resolution 728px",                     2.027, 0.4977, 0.3399, False),
    ("Resolution 518px",                     1.089, 0.4708, 0.2858, False),
    ("Resolution 266px",                     0.536, 0.4410, 0.2450, False),
    ("Dynamic Quantization (INT8)",          4.918, 0.4643, 0.4073, False),
    ("ONNX Runtime (CPU)",                   4.940, 0.4898, 0.4013, False),
    ("Hybrid (tissue@266, nuclei@728)",      1.712, 0.4410, 0.3399, False),
    ("Low-Rank SVD (r=0.25, no fine-tune)",  3.960, 0.0492, 0.2197, False),
    ("Structured Pruning (head only)",       4.476, 0.4342, 0.3649, False),
    ("Knowledge Distillation (8 epochs)",    0.619, 0.3861, 0.3038, False),
]

DOD_LATENCY = 1.0
COLOR_BASELINE = "#e34948"   # red (slot 8) — stands out from the rest
COLOR_METHOD = "#2a78d6"     # blue (slot 1)
COLOR_MUTED = "#898781"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e1e0d9"

# Numbered labels (not text) avoid collisions in the high-latency cluster —
# the legend with full names lives in a separate box (see anti-patterns.md:
# >4 converging series -> number/facet instead of stacking text labels).
METHOD_NUMBER = {name: i + 1 for i, (name, *_ ) in enumerate(RESULTS)}


def plot_panel(ax, y_key, title, ylabel):
    for name, latency, tissue_f1, nuclei_f1, is_baseline in RESULTS:
        y = tissue_f1 if y_key == "tissue" else nuclei_f1
        color = COLOR_BASELINE if is_baseline else COLOR_METHOD
        marker = "*" if is_baseline else "o"
        size = 260 if is_baseline else 110

        ax.scatter(
            [latency], [y], color=color, marker=marker, s=size,
            edgecolors="#fcfcfb", linewidths=1.5, zorder=3,
        )
        ax.annotate(
            str(METHOD_NUMBER[name]), (latency, y), ha="center", va="center",
            fontsize=7.5 if is_baseline else 8, fontweight="bold", color="#fcfcfb", zorder=4,
        )

    ax.axvline(DOD_LATENCY, color=COLOR_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(DOD_LATENCY, 0.02, " 1s\n DoD", color=COLOR_TEXT_SECONDARY, fontsize=8, va="bottom", ha="left")

    ax.set_title(title, fontsize=11, color="#0b0b0b")
    ax.set_xlabel("Latency (seconds, log scale)", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.6)
    ax.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def main():
    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax_tissue, ax_nuclei) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "Model Compression Study — Latency vs. Quality Trade-off (all methods)",
        fontsize=14, fontweight="bold", color="#0b0b0b",
    )

    plot_panel(ax_tissue, "tissue", "Tissue Segmentation Quality", "Tissue Macro F1")
    plot_panel(ax_nuclei, "nuclei", "Nuclei Segmentation Quality", "Nuclei Macro F1")

    key_lines = [f"{METHOD_NUMBER[name]}. {name}" for name, *_ in RESULTS]
    key_text = "   ".join(key_lines[:5]) + "\n" + "   ".join(key_lines[5:])
    fig.text(0.5, -0.06, key_text, ha="center", fontsize=8, color=COLOR_TEXT_SECONDARY)

    fig.text(
        0.5, -0.11,
        "Red star = baseline (DINOv2 ViT-S/14 backbone, FP32, 1036px). Numbered blue dots = compression/optimization method.  "
        "CPU-only inference, 41 validation images · 17/07/2026",
        ha="center", fontsize=8, color=COLOR_MUTED,
    )

    plt.tight_layout()
    save_path = VIS_DIR / "model_compression_final_comparison.png"
    os.makedirs(VIS_DIR, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    logger.info(f"Figure saved: {save_path}")

    logger.info("=" * 100)
    logger.info(f"{'Method':^38} | {'Latency':^10} | {'Tissue F1':^10} | {'Nuclei F1':^10} | {'DoD':^6}")
    logger.info("=" * 100)
    for name, latency, tissue_f1, nuclei_f1, _ in RESULTS:
        dod = "OK" if latency < DOD_LATENCY else "FAIL"
        logger.info(f"{name:^38} | {latency:^10.3f} | {tissue_f1:^10.4f} | {nuclei_f1:^10.4f} | {dod:^6}")


if __name__ == "__main__":
    main()

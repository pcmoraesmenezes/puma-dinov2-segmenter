"""'Corporate' / LinkedIn-ready version of the model compression study.

Curated to 4 methods (not all 10) to tell ONE story in 3 seconds: naive
latency optimization breaks the clinical metric; distillation doesn't.
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# name, latency, nuclei_f1, role  (role: "context" | "neutral" | "critical" | "good")
BASELINE_NUCLEI_F1 = 0.401
METHODS = [
    ("Baseline\n(current model)",       5.27, 0.401, "context"),
    ("ONNX Runtime\n(free optimization)", 4.94, 0.401, "neutral"),
    ("Lower resolution\n(naive speed-up)", 0.54, 0.245, "critical"),
    ("Knowledge Distillation\n(recommended)", 0.62, 0.304, "good"),
]

DOD_LATENCY = 1.0

COLOR_CONTEXT = "#c3c2b7"
COLOR_NEUTRAL = "#898781"
COLOR_CRITICAL = "#d03b3b"
COLOR_GOOD = "#0ca30c"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"

ROLE_COLOR = {"context": COLOR_CONTEXT, "neutral": COLOR_NEUTRAL, "critical": COLOR_CRITICAL, "good": COLOR_GOOD}


def main():
    plt.rcParams["font.family"] = "sans-serif"

    # ordena por latencia decrescente (barra mais lenta em cima, mais rapida embaixo)
    ordered = sorted(METHODS, key=lambda m: m[1], reverse=True)
    names = [m[0] for m in ordered]
    latencies = [m[1] for m in ordered]
    nuclei_f1s = [m[2] for m in ordered]
    colors = [ROLE_COLOR[m[3]] for m in ordered]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor("#fcfcfb")

    y_pos = range(len(ordered))
    bars = ax.barh(y_pos, latencies, height=0.58, color=colors, zorder=3)

    for i, (bar, latency, f1) in enumerate(zip(bars, latencies, nuclei_f1s)):
        ax.text(
            latency + 0.12, bar.get_y() + bar.get_height() / 2,
            f"{latency:.2f}s", va="center", ha="left",
            fontsize=13, fontweight="bold", color=COLOR_TEXT_PRIMARY,
        )
        retained_pct = 100 * f1 / BASELINE_NUCLEI_F1
        ax.text(
            latency + 0.12, bar.get_y() + bar.get_height() / 2 - 0.24,
            f"clinical accuracy retained: {retained_pct:.0f}%", va="center", ha="left",
            fontsize=9, color=COLOR_TEXT_SECONDARY, style="italic",
        )

    ax.axvline(DOD_LATENCY, color=COLOR_TEXT_SECONDARY, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.text(
        DOD_LATENCY, len(ordered) - 0.35, "  real-time\n  target",
        fontsize=9, color=COLOR_TEXT_SECONDARY, ha="left", va="top", fontweight="bold",
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=11, color=COLOR_TEXT_PRIMARY)
    ax.set_xlabel("Inference latency (seconds) — lower is better", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax.set_xlim(0, max(latencies) * 1.32)
    ax.invert_yaxis()

    ax.grid(True, axis="x", color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, left=False)

    fig.suptitle(
        "Cutting Medical AI Inference Time by 8.5× — With Minimal Impact on Clinical Accuracy",
        fontsize=17, fontweight="bold", color=COLOR_TEXT_PRIMARY, y=1.02, x=0.02, ha="left",
    )
    fig.text(
        0.02, 0.955,
        "Model compression study on a DINOv2-based tissue & nuclei segmentation model, self-hosted CPU deployment",
        fontsize=10.5, color=COLOR_TEXT_SECONDARY, ha="left",
    )

    # color -> meaning legend (green = recommended, red = cautionary, gray = reference)
    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=COLOR_GOOD, markeredgewidth=0, label="Recommended"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=COLOR_CRITICAL, markeredgewidth=0, label="Fast but hurts accuracy"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=COLOR_NEUTRAL, markeredgewidth=0, label="No meaningful gain"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=COLOR_CONTEXT, markeredgewidth=0, label="Current baseline"),
    ]
    legend = ax.legend(
        handles=legend_handles, loc="lower right", frameon=False, fontsize=9.5,
        bbox_to_anchor=(1.0, -0.02),
    )
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_SECONDARY)

    fig.text(
        0.02, -0.03,
        "Backbone: DINOv2 ViT-S/14 · Measured on 41 real validation images, CPU-only inference · 17/07/2026",
        fontsize=8.5, color=COLOR_MUTED, ha="left",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    save_path = VIS_DIR / "linkedin_compression_summary.png"
    os.makedirs(VIS_DIR, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    logger.info(f"Figure saved: {save_path}")


if __name__ == "__main__":
    main()

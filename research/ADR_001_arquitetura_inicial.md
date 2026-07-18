# ADR 001: Decoder Architecture Choice (PUMA)

**Status:** Decided
**Date:** 2026-05-06
**Owner:** Paulo Menezes (AI Architect)

## 1. Problem Context
The PUMA challenge requires solving two simultaneous tasks:
1. **Task 1 (Macro):** Semantic segmentation of tissue (Tumor, Stroma, etc).
2. **Task 2 (Micro):** Segmentation/detection of individual cell nuclei instances.

The available hardware is limited to **16GB of RAM**, which imposes a severe constraint on model size during training.

## 2. Term Glossary
To ensure absolute clarity, here are the pieces of our architecture:

*   **Backbone (DINOv2):** The pre-trained "brain" that looks at the image. It stays **Frozen** to save memory and leverage the Foundation Model's prior knowledge.
*   **Head:** The final part of the model that makes the decision and draws the result.
*   **Linear:** A single mathematical output layer. Fast, but limited in detail.
*   **Hybrid:** A small sequence of layers (3 to 4) that combines information from different levels of the Backbone.
*   **Mask2Former:** A massive and complex decoder architecture, the current SOTA standard.

## 3. Technical Comparison

| Attribute | Linear Heads | Mask2Former | Hybrid Head (Choice) |
| :--- | :--- | :--- | :--- |
| **Code Complexity** | Low | Very High | Medium |
| **Memory Usage (VRAM)** | Minimal (< 4GB) | Maximum (> 12GB) | Moderate (~6GB) |
| **Detail Resolution** | Low (Myopic) | Very High | High (Focused) |
| **Feasibility (16GB RAM)** | Full | Low/None | High |

## 4. Decision: Hybrid Architecture (Hybrid Head)
The decision is to implement a **Hybrid Head** connected to the **DINOv2 (ViT-S/14)** backbone.

**Rationale:**
1. **Micro Precision:** Nuclei segmentation in PUMA requires centroid precision (15px limit). A pure Linear head would struggle to reach the required resolution.
2. **Resource Efficiency:** The hybrid architecture allows extracting features from multiple DINOv2 blocks (multiscale) without the computational overhead of a full Transformer Decoder like Mask2Former.
3. **Scalability:** Allows evolving into a *Feature Caching* pipeline if direct training still proves too heavy.

## 5. Implications
* **Data Engineering:** The pipeline must prepare tissue masks and heatmaps for the nuclei.
* **Training:** We will only optimize the Head's weights, leaving DINOv2 untouched.
* **Verification:** We will validate the scale alignment between the Backbone (Patch Size 14) and the output Heads (1024x1024 resolution).

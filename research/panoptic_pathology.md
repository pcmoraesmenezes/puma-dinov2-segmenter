# 🔬 Panoptic Segmentation of nUclei and tissue in advanced MelanomA (PUMA)

## 📖 1. Clinical Context and Thesis (Background)

Melanoma is one of the most aggressive skin cancers. Treatment success, especially immunotherapy in advanced cases, depends on the host's immune response.

> [!IMPORTANT]
> **The PUMA Thesis:** the presence and, crucially, the **spatial location** of Tumor-Infiltrating Lymphocytes (TILs) are predictive biomarkers of survival. Manual scoring methods are subjective; AI steps in to bring objectivity and granularity (intratumoral vs. stromal).

---

## 📊 2. Dataset Specifications

The PUMA dataset is built on a multi-scale data infrastructure:

| Attribute | Technical Detail |
| :--- | :--- |
| **Volume** | 310 ROIs (155 Primary / 155 Metastatic) |
| **Main Resolution** | 1024 x 1024 pixels (40x magnification) |
| **Global Context** | 5120 x 5120 pixels (centered on the ROI) |
| **Labels** | GeoJSON (high-precision polygons) |
| **Validation** | Senior (board-certified) dermatopathologist |

---

## 🛠️ 3. Task Definitions (Tracks)

The challenge requires solving two Computer Vision problems of distinct nature simultaneously:

### 🟢 Task 1: Semantic Tissue Segmentation (Macro)
Segmentation of dense tissue regions.
- **Classes:** Stroma, Blood Vessels, Tumor, Epithelium (Epidermis), and Necrosis.
- **Background:** Class 0 (ignored in the metric).

### 🔵 Task 2: Nuclei Instance Segmentation (Micro)
Individual detection and segmentation of cell nuclei.

| Level | Instance Classes |
| :--- | :--- |
| **Track 1** | Tumor, TILs (Lymphocytes/Plasma), Other |
| **Track 2** | 10 detailed classes (Neutrophils, Endothelium, Apoptotic, etc.) |

---

## 📈 4. Evaluation Protocol (Success Metrics)

The Architect must optimize the model for two conflicting metrics:

### 📐 Macro-F1 Score (Task 2)
Based on a **geometric "hit"** criterion:
1. **Centroid extraction:** polygons are converted into points (x, y).
2. **Censoring radius (15px):** a prediction is only valid if its centroid is within 15 pixels of a ground truth point.
3. **Matching:** prioritized by `confidence_score` and proximity.
4. **Class alignment:** if a match occurs but the class differs, it's counted as a **False Positive (FP)**.

### 🎲 Micro-Dice Score (Task 1)
Intersection-over-union (IoU) computed for the tissue classes, concatenating all segmentation results before averaging.

---

## 🏗️ 5. Architectural Insights

> [!TIP]
> **DinoV2 Strategy:** since the backbone is *frozen*, the quality of the panoptic segmentation will depend entirely on the "Head's" ability to decode the ViT's features at two scales:
> 1. **Local (Nuclei):** requires high spatial resolution (patch size 14 is ideal).
> 2. **Global (Tissue):** requires wide receptive fields to understand the stroma vs. tumor architecture.

### Identified Engineering Challenges:
- **GeoJSON → Mask conversion:** needs efficient rasterization for training.
- **Duality Loss:** balancing a pixel-wise cross-entropy loss (Tissue) with an instance detection/segmentation loss (Nuclei).
- **Memory Management:** the pipeline must support the 5120x5120 (Context) volume on 16GB RAM setups.

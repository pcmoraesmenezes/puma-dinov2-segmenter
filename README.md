# DINOv2 for Panoptic Pathology 🔬

[![Challenge](https://img.shields.io/badge/Challenge-PUMA--MICCAI--2024-blue)](https://puma.grand-challenge.org/)
[![Model](https://img.shields.io/badge/Backbone-DINOv2--ViT--S14-green)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This repository hosts a technical implementation and study of **Panoptic Segmentation** in digital pathology (Melanoma) using **DINOv2**. The project explores the effectiveness of Self-Supervised Vision Transformers as universal feature extractors for complex clinical diagnostics.

---

## 🏗️ Project Overview

Traditional medical AI pipelines often rely on supervised pre-training on domain-specific data. This project shifts the paradigm by utilizing **DINOv2** (frozen backbone) to validate the "Universal Feature Extractor" hypothesis in the context of the **PUMA Challenge (MICCAI 2024)**.

The task involves simultaneous **Instance Segmentation** (cell nuclei) and **Semantic Segmentation** (tissue architecture), posing a significant challenge in multi-scale spatial reasoning.

### 🎯 Technical Objectives
*   **Feature Probing:** Quantify the generalizability of DINOv2's SSL features on H&E-stained histopathology images.
*   **Panoptic Head Implementation:** Design a lightweight architecture for dual-task segmentation (Micro + Macro) on top of frozen embeddings.
*   **Hardware-Aware MLOps:** Implement an **Offline Feature Caching** pipeline to enable high-resolution training on limited hardware (16GB RAM setups).

---

## 📊 Dataset: PUMA Challenge

We leverage the official **PUMA (Panoptic segmentation of nUclei and tissue in MelanomA)** dataset, which includes:
*   **High-Resolution ROIs:** 1024x1024 pixel patches.
*   **Context ROIs:** 5120x5120 pixel patches for global tissue architecture.
*   **Expert Annotations:** Fine-grained GeoJSON labels for nuclei (instance) and tissue (semantic) classes.


## 📜 Acknowledgments

*   **Meta AI** for the [DINOv2](https://github.com/facebookresearch/dinov2) foundation model.
*   **MICCAI 2024 / PUMA Challenge** organizers for the [dataset and clinical annotations](https://zenodo.org/records/14869398).

---

*Developed as part of an AI Architecture seniority study (O1).*

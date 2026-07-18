# ARCHITECTURE: DINOv2 for Panoptic Pathology (PUMA)

Technical documentation for the project — architecture decisions, pipeline, and trade-offs.

---

## 1. Clinical Problem

Advanced melanoma is a skin cancer that, at an advanced stage, has metastasized — spreading to other organs. The main treatment in these cases is immunotherapy, which amplifies the patient's own immune response against the tumor.

The success of immunotherapy depends directly on the presence and location of **TILs** (Tumor-Infiltrating Lymphocytes) — defense cells that have infiltrated the tumor tissue. TILs inside the tumor (intratumoral) indicate a different prognosis than TILs around it, in the stroma (peritumoral).

Manual assessment by pathologists is slow and suffers from inter-observer variation: two doctors analyzing the same slide can reach different scores. AI solves this with objectivity, speed, and granularity — automatically identifying the location and density of TILs at scale.

---

## 2. Dataset (PUMA)

The PUMA dataset is a computer vision dataset built and validated by a dermatopathologist to identify advanced melanoma. The images are `.tif` files and the annotations are in GeoJSON format — high-precision polygons drawn over the biological structures.

The dataset contains two types of annotation:
- **Tissue**: semantic regions (Tumor, Stroma, Blood Vessel, Epidermis, Necrosis)
- **Nuclei**: individual cell instances at two levels — Track 1 (Tumor, TILs, Other) and Track 2 (10 detailed classes)

For use in deep learning models, the GeoJSON annotations need to go through **rasterization** — converting the polygons into pixel masks with numeric labels.

**Volume:** 310 ROIs (155 primary + 155 metastatic)
**ROI Resolution:** 1024×1024 pixels (40x magnification)
**Context Resolution:** 5120×5120 pixels

---

## 3. Architecture Decisions

**Why DINOv2 instead of a ResNet trained from scratch?**
The PUMA dataset has only 310 ROIs — insufficient to train a backbone from scratch without severe overfitting. DINOv2, pre-trained via self-supervised learning on 142 million images, provides rich, generalizable features without requiring domain data. The project tests the hypothesis that these features transfer to H&E medical images the model never saw during pre-training.

**Why was the backbone kept frozen?**
Two combined reasons: hardware limitation (16GB RAM) and a small dataset. Fine-tuning DINOv2 would require gradients for ~21M parameters — infeasible on the available machine — and on 310 samples it would cause overfitting. The frozen backbone also enables offline feature caching, removing DINOv2 from the training loop.

**Why a Hybrid Head instead of Linear or Mask2Former?**
The Linear Head is computationally simple but lacks the capacity to learn the transformations needed for precise pixel-level predictions — especially for nuclei, with a 15px radius criterion. Mask2Former is the SOTA standard but requires more than 12GB of VRAM, infeasible on the available hardware. The Hybrid Head balances the two: enough refinement capacity at a moderate computational cost.

---

## 4. Pipeline

| Script | Responsibility |
|---|---|
| `01_data_visualization.py` | Visual diagnostics: renders the `.tif` images with the GeoJSON polygon overlay via Matplotlib. Used to validate image/annotation alignment before training. |
| `02_feature_extractor.py` | Does two things: (1) runs DINOv2 on each image and saves the features as `.pt` on disk — the offline cache; (2) rasterizes the GeoJSONs into PNG masks, assigning numeric labels to each pixel. |
| `03_dataset.py` | PyTorch Dataset that loads the cached features and masks. Supports RAM caching to remove disk I/O during training. |
| `04_model.py` | Defines the architecture: frozen DINOv2 backbone + two independent heads (tissue and nuclei). Each head refines the `(B, 384, 74, 74)` features via convolutions down to `(B, 128, 74, 74)`, upsamples to `(B, 128, 1024, 1024)`, and projects onto the final classes via a 1×1 Conv2d. |
| `05_train.py` | Training loop: reproducible 80/20 split, AdamW optimizer, dual CrossEntropy loss (tissue + nuclei), per-epoch metrics (accuracy and macro F1), automatic checkpoint of the best model. Trains only the heads' weights — the backbone stays frozen. |
| `06_inference.py` | Runs the trained model over validation samples and saves side-by-side visualizations (original image + tissue overlay + nuclei overlay). |

`config.py` (root) centralizes all the project's paths (dataset, checkpoints, visualizations) via environment variables with repo-relative defaults — used by `pipeline/`, `serving/`, and `optimization/`.

### Serving (deploy) — production code only

Reorganized on 2026-07-17: `serving/` holds only the code the production API depends on directly; research/benchmark scripts live in `optimization/` (see Section 7).

| Module | Responsibility |
|---|---|
| `serving/student_model.py` | `StudentBackbone` — the distilled backbone architecture (CNN, 1.3M params). Lives here because production depends on it directly; `optimization/distillation.py` imports it from here to train it, not the other way around. |
| `serving/preprocessing.py` | Shared image transform (resize + normalize) — same preprocessing for serving and for the research scripts, single source of truth. |
| `serving/load_model.py` | Loads the backbone (DINOv2 ViT-S/14, frozen) + head (`DinoPUMASegmenter`, from checkpoint) pair and validates with a dummy forward pass (healthcheck). |
| `serving/api.py` | FastAPI app — `/predict` (image in, tissue+nuclei masks out), `/health`, `/metrics`. Serves the distilled student, not the original DINOv2 (see Section 7 — it's the only combination that meets the latency DoD). |
| `serving/test_api.py` | Integration tests (pytest + TestClient) for the three routes. |

**Measured risk (2026-07-17):** a full forward pass (backbone + head) on CPU takes ~3.4s — above the deploy epic's p95 < 1s latency criterion. The target server (self-hosted) is also CPU-only. Model compression (quantization/pruning/distillation) is a prerequisite, not a future optimization — see the Deploy DinoV2 KR in the OKR Hub. **Resolved:** see Section 7 — the API serves the distilled student.

---

## 5. Trade-offs and Limitations

**Training vs. inference resolution**
Training happens at 74×74 (DINOv2's feature resolution) instead of 1024×1024. Each grid cell represents ~14px of the original image. This saves ~190x the processing but reduces localization precision — especially critical for small nuclei, where the challenge's 15px margin fits inside a single feature-map cell.

**Instance segmentation simplified to semantic**
All nuclei of the same type get the same numeric label. The model learns "there are lymphocytes here" but doesn't distinguish lymphocyte #1 from #47. Producing individual centroids (required by the challenge) would need post-processing via connected components on the predicted mask.

**Small dataset**
310 ROIs is a small volume for robust generalization. The model may not perform well on slides from labs with different H&E staining protocols (concentration variation produces pinker or bluer images).

---

## 6. Results

Trained on 205 ROIs, 80/20 split (164 train / 41 validation), frozen backbone, AdamW with lr=5e-4 and weight_decay=1e-3.

| Epoch | Train Loss | Val Loss | Tissue Acc | Tissue F1 | Nuclei Acc | Nuclei F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.8992 | 1.3223 | 0.7749 | 0.4179 | 0.8222 | 0.5155 |
| 4 | 0.9454 | 0.9714 | 0.8041 | 0.4488 | 0.8527 | 0.5881 |
| **8 (best)** | **0.7607** | **0.8826** | **0.8020** | **0.4288** | **0.8621** | **0.6113** |
| 13 | 0.6786 | 0.9303 | 0.7746 | 0.4462 | 0.8551 | 0.6360 |
| 15 | 0.5918 | 0.9636 | 0.7765 | 0.4186 | 0.8652 | 0.6344 |

**Best checkpoint:** epoch 8, val loss 0.8826

**Observations:**
- Lowering the LR (1e-3 → 5e-4) let the model keep improving past epoch 3 — a problem observed in the previous training run.
- Nuclei F1 (≈0.61) consistently beats Tissue F1 (≈0.43). Rare tissue classes like Necrosis and Blood Vessel drag the F1 average down.
- From epoch 8 onward, train loss keeps dropping while val loss oscillates — a sign of mild overfitting. An LR scheduler or early stopping would improve the result.
- Full learning curves: `visualizations/training_history_dinov2.png`

---

## 7. Model Compression Study (2026-07-17)

Practical validation of chapter 7 ("Model Deployment and Prediction Serving") of *Designing Machine Learning Systems* (Chip Huyen), motivated by the latency gate of the self-hosted deploy epic (CPU-only server, no GPU). Six techniques tested under the same protocol — 41 real validation images, macro F1 — against the baseline (DINOv2 ViT-S/14 backbone, FP32, 1036px).

**Figure:** `visualizations/model_compression_final_comparison.png` · **Scripts:** `optimization/eval_*.py`, `optimization/resolution_tradeoff.py`, `optimization/distillation.py` (separate folder from `serving/`, reorganized 2026-07-17 — research/experimentation doesn't mix with production code).

| Method | Latency | Tissue F1 | Nuclei F1 | DoD (<1s) |
|---|---|---|---|---|
| Baseline (1036px, FP32) | 5.27s | 0.490 | 0.401 | ❌ |
| Dynamic Quantization (INT8) | 4.92s | 0.464 | 0.407 | ❌ |
| ONNX Runtime (CPU) | 4.94s | 0.490 | 0.401 | ❌ |
| Resolution 728px | 2.03s | 0.498 | 0.340 | ❌ |
| Hybrid (tissue@266, nuclei@728) | 1.71s | 0.441 | 0.340 | ❌ |
| Resolution 518px | 1.09s | 0.471 | 0.286 | ❌ |
| Structured Pruning (head only) | 4.48s | 0.434 | 0.365 | ❌ |
| Low-Rank SVD (r=0.25, no fine-tune) | 3.96s | 0.049 | 0.220 | ❌ |
| Resolution 266px | 0.54s | 0.441 | 0.245 | ✅ |
| **Knowledge Distillation (student CNN, 8 epochs)** | **0.62s** | 0.386 | 0.304 | ✅ |

**Conclusion:** only two techniques meet the latency DoD (<1s) — resolution 266px and knowledge distillation. Distillation preserves the clinically more critical metric (nuclei/TILs, 15px margin) better than plain resolution reduction, at the cost of a slightly higher latency (still well within the DoD). It's the most solid path to production, but needs more training epochs and possibly joint fine-tuning of the head before going to real production.

**Meta-conclusion:** the theoretical risk hierarchy from chapter 7 (quantization = safe, pruning = medium, distillation = risky) did not hold up empirically here — distillation was the technique that best balanced latency and accuracy, while low-rank (similar theoretical risk) was the most destructive one tested, and the "safe" techniques (quantization, ONNX) barely moved the latency on real hardware. Theory gives you the right prior on what to try first; only measuring on the real hardware and model tells you what actually works.

**Important methodological note — this study's F1 is not comparable to Section 6's F1:** the original validation (`pipeline/05_train.py`, `validate()`) measures F1 with `upscale=False`, comparing the prediction against the ground-truth mask **downsampled to 74×74** (the features' native resolution). This compression study measures with `upscale=True`, comparing against the ground-truth mask at its real resolution (1024×1024) — the real deployment scenario. These are different metrics, not a regression: the 74×74 version "forgives" small pixel errors (smoothed out by the downsampling), which is especially critical for small nuclei; the 1024×1024 version is harsher and more realistic. That's why the baseline here reports Nuclei F1 0.401 instead of the ~0.61-0.63 documented in Section 6 — same model, a different and stricter measuring stick.

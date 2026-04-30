# Ejection Fraction Estimation Using an R(2+1)D-18 CNN

Year 2 undergraduate group project investigating automated left ventricular ejection fraction (EF) estimation from echocardiogram video using a spatiotemporal convolutional neural network.

This work reproduces and extends the EchoNet-Dynamic pipeline (Ouyang et al., 2020) using a pretrained R(2+1)D-18 backbone (Tran et al., 2018), adapted for single-channel grayscale input and augmented with a dual-task shared-head architecture, post-hoc clinical calibration, and Grad-CAM saliency analysis.

The accompanying report is included in this repository: [Ejection Fraction Estimation Using an R(2+1)D-18 Convoluted Neural Network.docx](Ejection%20Fraction%20Estimation%20Using%20an%20R(2%2B1)D-18%20Convoluted%20Neural%20Network.docx)

---

## Overview

Ejection fraction — the percentage of blood pumped from the left ventricle per beat — is a critical clinical biomarker for heart failure diagnosis. Manual estimation from echocardiography is time-consuming and subject to inter-observer variability. This project explores whether a spatiotemporal deep learning model can replicate cardiologist-level EF estimation reliably.

**Task:** Simultaneous EF regression (continuous value) and classification into three clinical categories:

| Category | EF Range |
|---|---|
| Reduced | < 40% |
| Mildly Reduced | 40 – 49% |
| Preserved | ≥ 50% |

---

## Architecture

The model is built on an **R(2+1)D-18** backbone pretrained on Kinetics-400. The original RGB input convolution is adapted for single-channel echocardiogram input by averaging the pretrained weight tensor across the colour dimension, preserving Kinetics-400 representations without discarding any weights.

A **shared MLP embedding** (512 → 256 → 128) feeds two task-specific heads in parallel:

- `regression_head` — scalar EF prediction
- `classification_head` — 3-class logits

This dual-head design extends the original Stanford single-head approach and is evaluated against ablated variants.

```
Input (B, 1, T, H, W)
       │
  R(2+1)D-18 backbone
       │
  Shared MLP (512→256→128)
     ┌──┴──┐
  Regression  Classification
   Head (1)    Head (3)
```

---

## Key Extensions Beyond the Original Paper

- **Dual-task shared head** — joint regression and classification from a shared embedding
- **Grayscale weight averaging** — preserves Kinetics-400 pretraining for single-channel input
- **Post-hoc clinical calibration** — test-time augmentation (10 clips), temperature scaling, and recall-optimised classification thresholds
- **Grad-CAM saliency analysis** — temporal activation maps with left-ventricular contour overlay and IoU evaluation against VolumeTracings annotations
- **Ablation study** — four controlled variants: baseline (16-frame), no pretraining, no shared head, regression-only
- **Per-EF-band subgroup analysis** — disaggregated performance across the EF distribution
- **95% bootstrap confidence intervals** — on all reported metrics

---

## Repository Structure

```
├── main.py                  # Full training + evaluation pipeline entry point
├── app/
│   └── start.py             # customtkinter GUI for single-video inference
├── src/
│   ├── CNN_model.py         # R(2+1)D-18 dual-head architecture
│   ├── config.py            # All hyperparameters and paths
│   ├── data_processing.py   # EchoNet-Dynamic dataloader and augmentation
│   ├── training.py          # Training loop, optimiser, LR scheduler
│   └── analysis.py          # Evaluation, Grad-CAM, ablation, subgroup analysis
├── artifacts/
│   ├── best_r2plus1d.pt     # Best model checkpoint
│   ├── dataset_statistics.json
│   ├── ablation/            # Ablation variant checkpoints
│   ├── evaluation/          # Test set predictions CSV
│   └── gradcam/             # Saved Grad-CAM visualisations
├── dataset/                 # EchoNet-Dynamic dataset (not included — see below)
└── setup/
    ├── requirements.txt
    ├── setup.bat            # Windows setup script
    └── setup.sh             # Linux/macOS setup script
```

---

## Setup

### 1. Dataset

Download the **EchoNet-Dynamic** dataset from Kaggle and place the contents inside the `dataset/` directory:

> https://www.kaggle.com/datasets/mahnurrahman/echonet-dynamic

Expected structure:
```
dataset/
├── FileList.csv
├── VolumeTracings.csv
└── Videos/
```

If your dataset lives elsewhere, create a `local_config.json` in the project root:
```json
{ "DATASET_PATH": "/path/to/your/dataset" }
```

### 2. Environment

**Windows:**
```bat
setup\setup.bat
```

**Linux / macOS:**
```bash
chmod +x setup/setup.sh
./setup/setup.sh
```

This creates a virtual environment and installs all dependencies. PyTorch is listed for CUDA 12.8 — adjust the index URL in `requirements.txt` for your CUDA version if needed.

### 3. Run

**Full pipeline** (training + evaluation + Grad-CAM + ablation):
```bash
python main.py
```

Training is skipped automatically if `artifacts/best_r2plus1d.pt` already exists.

**GUI inference app:**
```bash
python app/start.py
```

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Backbone | R(2+1)D-18 | Kinetics-400 pretrained |
| Input frames | 32 | Sampled every 2nd frame |
| Frame size | 112 × 112 | |
| Batch size | 16 | |
| Optimiser | SGD | Following Ouyang et al. |
| Learning rate | 1e-4 | Linear scaling rule applied |
| Epochs | 45 | Early stopping patience = 10 |
| Dropout | 0.5 | In shared MLP |

---

## Citation / Reference

This work builds on:

- Ouyang, D. et al. (2020). *Video-based AI for beat-to-beat assessment of cardiac function*. Nature, 580, 252–256.
- Tran, D. et al. (2018). *A Closer Look at Spatiotemporal Convolutions for Action Recognition*. CVPR.

---

## Notes

This was a group undergraduate coursework submission. Contributions from all group members are reflected in the git history.

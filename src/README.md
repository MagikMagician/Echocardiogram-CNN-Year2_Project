# src Module README

### config.py
- Central project settings (paths, frame size, batch size, workers, learning settings).
- EF category thresholds:
  - Reduced: EF < 40
  - Mildly Reduced: 40 <= EF < 50
  - Preserved: EF >= 50
- Supports local path override via `local_config.json`.

### data_processing.py
- Validates dataset paths and expected CSV columns (`FileName`, `Split`, `EF`).
- Loads TRAIN/VAL/TEST video paths and labels.
- Extracts uniformly sampled grayscale frames from each video.
- Resizes frames to target shape and pads short videos with zeros.
- Computes dataset mean/std (with cache at `artifacts/dataset_statistics.json`).
- Builds `EchoDataset` that returns:
  - video tensor: `(1, T, H, W)`
  - EF continuous label
  - EF category label (0/1/2)
- Creates train/val/test DataLoaders with consistent settings.

## CNN + Training (CNN_model.py + training.py)

### CNN_model.py
- Defines a 3D CNN for echocardiogram video volumes.
- Uses stacked Conv3D blocks for temporal-spatial feature extraction.
- Applies adaptive pooling to support variable frame and image sizes.
- Produces dual outputs: EF regression value and class logits.

### training.py
- Initializes device, model, losses, optimizer, and LR scheduler.
- Runs train/validation epochs with weighted multi-task loss.
- Tracks loss, MAE, and classification accuracy per epoch.
- Saves the best checkpoint and uses early stopping.


## Analysis (analysis.py)


## Minimal Progress Checklist
- [x] Data loading + preprocessing
- [x] Dataset + DataLoaders
- [x] CNN architecture
- [x] Training loop
- [ ] Test evaluation
- [ ] Grad-CAM
- [ ] Ablation study

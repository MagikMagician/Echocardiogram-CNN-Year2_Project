# Core Deep Learning
import torch
from pathlib import Path

from src.data_processing import create_dataloaders
from src.training import initialize_training_components, train_model
from src.analysis import (
    evaluate_on_test_set,
    plot_training_curves,
    run_gradcam_visualization,
    run_gradcam_with_lv_overlay,
    run_ablation_study,
    run_gradcam_lv_iou,
    run_subgroup_analysis,
    run_calibration_analysis,
)

CHECKPOINT = Path("artifacts/best_r2plus1d.pt")

# =============================================================================
# EchoNet-Dynamic — automated ejection fraction estimation
#
# Predicts continuous EF values (regression) and classifies into three clinical
# categories: Reduced (<40 %), Mildly Reduced (40–49 %), Preserved (≥50 %).
# Dataset: EchoNet-Dynamic (Stanford University).
# =============================================================================


def run_pipeline() -> None:
    """Run the full training and evaluation pipeline end-to-end."""
    # Build dataloaders — training augmentation applied inside create_dataloaders.
    train_loader, val_loader, _, class_weights = create_dataloaders()

    print("Data pipeline initialized.")
    print(f"  GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")

    # Build the R(2+1)D model, SGD optimiser, and learning-rate scheduler.
    components = initialize_training_components(class_weights=class_weights)

    # Skip training if a checkpoint already exists (re-run evaluation only).
    if CHECKPOINT.exists():
        print(f"Checkpoint found at {CHECKPOINT} — skipping training.")
    else:
        history = train_model(train_loader, val_loader, components)
        print(f"Training complete. Epochs run: {len(history['train_loss'])}")
        plot_training_curves(history)

    # Evaluate the best checkpoint on the held-out test set.
    evaluate_on_test_set()

    # Generate Grad-CAM overlays for a sample of test videos.
    run_gradcam_visualization(num_samples=8)

    # Generate layered Grad-CAM figures: frame / expert LV contour / model attention.
    run_gradcam_with_lv_overlay(num_samples=8)

    # Run ablation variants to quantify the effect of each design choice.
    run_ablation_study()

    # ── Novel contributions beyond Stanford EchoNet-Dynamic ──────────────────
    # 1. Grad-CAM localisation vs expert LV annotations (IoU)
    run_gradcam_lv_iou()

    # 2. Subgroup analysis: per-EF-band metrics + clinical cost confusion matrix
    run_subgroup_analysis()

    # 3. Calibration: reliability diagram + Expected Calibration Error (ECE)
    run_calibration_analysis()

if __name__ == "__main__":
    try:
        run_pipeline()

    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please update Config.DATASET_PATH in data_processing.py")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

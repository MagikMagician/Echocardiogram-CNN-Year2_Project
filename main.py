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
    run_clinical_evaluation,
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

    # ── Post-hoc clinical evaluation ─────────────────────────────────────────
    # Applies TTA, temperature scaling, and recall-optimised thresholds.
    # Returns post-hoc predictions used by all downstream evaluation functions
    # so every output chart reflects the improved model, not the raw baseline.
    post_hoc = run_clinical_evaluation(n_tta_clips=10)

    # Regression and classification metrics using post-hoc TTA predictions.
    evaluate_on_test_set(post_hoc=post_hoc)

    # Per-EF-band subgroup analysis using post-hoc predictions.
    run_subgroup_analysis(post_hoc=post_hoc)

    # ── Grad-CAM visualisations ───────────────────────────────────────────────
    run_gradcam_visualization(num_samples=8)
    run_gradcam_with_lv_overlay(num_samples=8)
    run_gradcam_lv_iou()

    # ── Ablation study ────────────────────────────────────────────────────────
    run_ablation_study()

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

# Core Deep Learning
import torch
from pathlib import Path

# Data, Training, and Analysis Modules
from src.data_processing import create_dataloaders
from src.training import initialize_training_components, train_model
from src.analysis import (
    evaluate_on_test_set,
    plot_training_curves,
    run_gradcam_visualization,
    run_ablation_study,
)

CHECKPOINT = Path("artifacts/best_r2plus1d.pt")

# ==============================================================================
# PROJECT: Automated EF Estimation from Echocardiogram Videos
# Dataset: EchoNet-Dynamic (Stanford University)
# Goal: Predict continuous EF values (regression) and classify into clinical categories
# Categories: Reduced (<40%), Mildly Reduced (40-49%), Preserved (≥50%)
# ==============================================================================


def run_pipeline() -> None:
    """
    Run the end-to-end project pipeline by orchestrating all module steps.
    """
    # STEP 4: CREATE DATALOADERS
    train_loader, val_loader, _, class_weights = create_dataloaders()

    print("Data pipeline initialized.")
    print(f"  GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")

    # STEP 5: INITIALIZE MODEL, LOSS, OPTIMIZER
    components = initialize_training_components(class_weights=class_weights)

    # STEP 6: TRAINING LOOP — skip if checkpoint already exists
    if CHECKPOINT.exists():
        print(f"Checkpoint found at {CHECKPOINT} — skipping training.")
    else:
        history = train_model(train_loader, val_loader, components)
        print(f"Training complete. Epochs run: {len(history['train_loss'])}")
        plot_training_curves(history)

    # STEP 7: EVALUATION ON TEST SET
    evaluate_on_test_set()

    # STEP 8: GRAD-CAM VISUALIZATION 
    run_gradcam_visualization()

    # STEP 9: ABLATION STUDY 
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

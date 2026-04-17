from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.CNN_model import CNN3D
from src.config import Config

config = Config()

EpochMetrics = Dict[str, float]
TrainingHistory = Dict[str, List[float]]


@dataclass
class TrainingComponents:
    """Bundle model and optimization objects used during training."""

    device: torch.device
    model: CNN3D
    regression_loss_fn: nn.Module
    classification_loss_fn: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau]


def _empty_epoch_metrics() -> EpochMetrics:
    """Return zeroed metrics for empty dataloaders."""
    return {
        "loss": 0.0,
        "regression_loss": 0.0,
        "classification_loss": 0.0,
        "mae": 0.0,
        "accuracy": 0.0,
    }


# ==============================================================================
# STEP 5: INITIALIZE MODEL, LOSS, OPTIMIZER
# ==============================================================================


def initialize_training_components(
    learning_rate: float = config.LEARNING_RATE,
    num_classes: int = config.NUM_CATEGORIES,
) -> TrainingComponents:
    """Initialize model, losses, optimizer, and scheduler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN3D(num_classes=num_classes).to(device)
    regression_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Reduce learning rate when validation loss plateaus.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    return TrainingComponents(
        device=device,
        model=model,
        regression_loss_fn=regression_loss_fn,
        classification_loss_fn=classification_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )


# ==============================================================================
# STEP 6: TRAINING LOOP
# ==============================================================================


def _run_epoch(
    loader: DataLoader,
    components: TrainingComponents,
    regression_weight: float,
    classification_weight: float,
    is_training: bool,
) -> EpochMetrics:
    """Run one train or validation epoch and return averaged metrics."""
    model = components.model
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_regression_loss = 0.0
    total_classification_loss = 0.0
    total_mae = 0.0
    total_correct = 0
    total_samples = 0

    for videos, ef_values, ef_classes in loader:
        # Keep tensors on the same device as the model for fast forward/backward passes.
        videos = videos.to(components.device, non_blocking=True)
        ef_values = ef_values.to(components.device, non_blocking=True)
        ef_classes = ef_classes.to(components.device, non_blocking=True)

        if is_training:
            components.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            ef_regression, ef_logits = model(videos)
            regression_loss = components.regression_loss_fn(ef_regression, ef_values)
            classification_loss = components.classification_loss_fn(ef_logits, ef_classes)
            loss = (regression_weight * regression_loss) + (
                classification_weight * classification_loss
            )

            if is_training:
                loss.backward()
                components.optimizer.step()

        batch_size = videos.size(0)
        # Aggregate weighted by batch size so epoch averages are correct.
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_regression_loss += regression_loss.item() * batch_size
        total_classification_loss += classification_loss.item() * batch_size
        total_mae += torch.abs(ef_regression - ef_values).sum().item()
        predicted_classes = ef_logits.argmax(dim=1)
        total_correct += (predicted_classes == ef_classes).sum().item()

    if total_samples == 0:
        return _empty_epoch_metrics()

    return {
        "loss": total_loss / total_samples,
        "regression_loss": total_regression_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "mae": total_mae / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    components: TrainingComponents,
    num_epochs: int = config.NUM_EPOCHS,
    regression_weight: float = 1.0,
    classification_weight: float = 1.0,
    early_stopping_patience: int = config.PATIENCE,
    checkpoint_path: Path = Path("artifacts/best_cnn3d.pt"),
) -> TrainingHistory:
    """Train the model with validation and simple early stopping."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: TrainingHistory = {
        "train_loss": [],
        "train_mae": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_mae": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # Train first, then validate with the same metric pipeline.
        train_metrics = _run_epoch(
            loader=train_loader,
            components=components,
            regression_weight=regression_weight,
            classification_weight=classification_weight,
            is_training=True,
        )
        val_metrics = _run_epoch(
            loader=val_loader,
            components=components,
            regression_weight=regression_weight,
            classification_weight=classification_weight,
            is_training=False,
        )

        if components.scheduler is not None:
            # Scheduler tracks validation loss, not training loss.
            components.scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            # Save the best model seen so far for later evaluation.
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": components.model.state_dict(),
                    "optimizer_state_dict": components.optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        # Stop early when validation loss stops improving.
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(best val loss: {best_val_loss:.4f})."
            )
            break

    return history

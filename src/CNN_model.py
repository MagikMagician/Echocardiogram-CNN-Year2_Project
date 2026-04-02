# Core Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data Processing
from typing import Tuple

# ==============================================================================
# STEP 3: 3D CNN MODEL - Processes video volumes for EF estimation
# ==============================================================================
class CNN3D(nn.Module):
    def __init__(self, num_classes: int = 3):
        """
        3D CNN for EF estimation from echocardiogram videos.
        Architecture inspired by EchoNet-Dynamic paper (Ouyang et al., 2020).
        Args:
            num_classes: Number of EF categories (3: Reduced, Mildly Reduced, Preserved)
        """
        super(CNN3D, self).__init__()
        # TODO: Define Conv3d layers for spatial-temporal feature extraction
        # Example: self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3,3,3), padding=1)
        # TODO: Define BatchNorm3d for normalization
        # TODO: Define MaxPool3d layers for downsampling
        # TODO: Define fully connected layers for regression and classification
        # TODO: Regression head: outputs continuous EF value
        # TODO: Classification head: outputs 3 class probabilities
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input x shape: (batch, 1, time, height, width)
        # TODO: Pass through conv3d + batchnorm + relu + maxpool layers
        # TODO: Flatten spatial-temporal features
        # TODO: Pass through fully connected layers
        # TODO: Regression output: single EF value
        # TODO: Classification output: 3-class probabilities (softmax)
        # TODO: Return (ef_regression, ef_classification)
        pass


# ==============================================================================
# STEP 5: INITIALIZE MODEL, LOSS, OPTIMIZER
# ==============================================================================
# TODO: Set device (cuda if available, else cpu)
# TODO: Initialize CNN3D model and move to device
# TODO: Define regression loss: MSELoss for continuous EF prediction
# TODO: Define classification loss: CrossEntropyLoss for clinical categories
# TODO: Define optimizer: Adam with appropriate learning rate
# TODO: Optional: Define learning rate scheduler


# ==============================================================================
# STEP 6: TRAINING LOOP
# ==============================================================================
# TODO: Loop over epochs
#   TODO: Set model to train mode
#   TODO: Loop over batches in train_loader
#       TODO: Move data to device
#       TODO: Forward pass: get regression and classification outputs
#       TODO: Calculate regression loss (MSE)
#       TODO: Calculate classification loss (CrossEntropy)
#       TODO: Combine losses (weighted sum)
#       TODO: Backward pass and optimizer step
#       TODO: Track training metrics (loss, MAE)
#   TODO: Validation loop
#       TODO: Set model to eval mode
#       TODO: Calculate validation losses and metrics
#       TODO: Calculate MAE, MSE, R² for regression
#       TODO: Calculate accuracy, precision, recall, F1 for classification
#   TODO: Save best model based on validation performance
#   TODO: Print epoch results and plot loss curves


def initialize_training_components() -> None:
    """Placeholder for model/loss/optimizer setup implementation."""
    # TODO: Implement model initialization, losses, optimizer, and scheduler setup.
    pass


def train_model() -> None:
    """Placeholder for training loop implementation."""
    # TODO: Implement epoch and batch training loops with validation.
    pass

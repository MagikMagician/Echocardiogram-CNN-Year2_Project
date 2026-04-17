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

        def conv_block(
            in_channels: int,
            out_channels: int,
            pool_kernel: Tuple[int, int, int],
        ) -> nn.Sequential:
            """Create one convolutional block for spatiotemporal feature extraction."""
            return nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3, 3),
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=pool_kernel),
            )

        self.feature_extractor = nn.Sequential(
            conv_block(1, 32, pool_kernel=(1, 2, 2)),
            conv_block(32, 64, pool_kernel=(2, 2, 2)),
            conv_block(64, 128, pool_kernel=(2, 2, 2)),
        )

        # Adaptive pooling keeps the model compatible with different frame counts/resolutions.
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self.regression_head = nn.Linear(64, 1)
        self.classification_head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return continuous EF prediction and classification logits."""
        if x.ndim != 5:
            raise ValueError(
                "Expected input shape (batch, 1, time, height, width), "
                f"received {tuple(x.shape)}"
            )

        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        embedding = self.shared_head(pooled)

        ef_regression = self.regression_head(embedding).squeeze(-1)
        # Return logits for CrossEntropyLoss; apply softmax only for reporting/inference.
        ef_classification = self.classification_head(embedding)

        return ef_regression, ef_classification


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

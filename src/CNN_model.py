# Core Deep Learning
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Typing
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
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 for classification.")

        super().__init__()

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
            # Keep temporal resolution in the first block, then downsample.
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

        # Task-specific heads from the same shared representation.
        self.regression_head = nn.Linear(64, 1)
        self.classification_head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return continuous EF prediction and classification logits."""
        if x.ndim != 5:
            raise ValueError(
                "Expected input shape (batch, 1, time, height, width), "
                f"received {tuple(x.shape)}"
            )
        if x.size(1) != 1:
            raise ValueError(
                "Expected a single-channel grayscale input at dimension 1, "
                f"received {x.size(1)} channels"
            )

        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        embedding = self.shared_head(pooled)

        # Match regression target shape: (batch,).
        ef_regression = self.regression_head(embedding).squeeze(-1)
        # CrossEntropyLoss expects raw logits.
        ef_classification = self.classification_head(embedding)

        return ef_regression, ef_classification


# ==============================================================================
# STEP 3b: R(2+1)D MODEL — EchoNet-Dynamic paper replication
# ==============================================================================
class R2Plus1DEF(nn.Module):
    """
    R(2+1)D-18 backbone (Tran et al., 2018) pre-trained on Kinetics-400.
    Adapts the RGB model for single-channel echocardiogram input by averaging
    the first-layer weights across the colour dimension.

    Two task-specific heads share the same temporal embedding:
        - regression_head  — outputs a scalar EF in R
        - classification_head — outputs C logits for EF category prediction
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()

        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r2plus1d_18(weights=weights)

        # Adapt first conv layer: 3-channel RGB -> 1-channel grayscale
        old_conv = backbone.stem[0]
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.stem[0] = new_conv

        # Strip the original FC classifier; keep feature layers only.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.regression_head = nn.Linear(128, 1)
        self.classification_head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, T, H, W) normalised grayscale video tensor

        Returns:
            ef_regression:     (B,)    continuous EF prediction
            ef_classification: (B, C)  raw logits for EF category
        """
        if x.ndim != 5 or x.size(1) != 1:
            raise ValueError(
                f"Expected single-channel input, received {x.size(1)} channels."
            )

        features = self.backbone(x)           # (B, 512)
        embedding = self.shared_head(features)
        ef_regression = self.regression_head(embedding).squeeze(-1)  # (B,)
        ef_classification = self.classification_head(embedding)       # (B, C)

        return ef_regression, ef_classification

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from typing import Tuple
class R2Plus1DEF(nn.Module):
    """
    R(2+1)D-18 backbone (Tran et al., 2018) pre-trained on Kinetics-400.
    Adapts the RGB model for single-channel echocardiogram input by averaging
    the first-layer weights across the colour dimension.

    Two task-specific heads share the same temporal embedding:
        - regression_head  — outputs a scalar EF in R
        - classification_head — outputs C logits for EF category prediction
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        use_shared_head: bool = True,
    ) -> None:
        super().__init__()

        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r2plus1d_18(weights=weights)

        # The pretrained backbone expects 3-channel RGB input. We average the
        # three input weight slices into one so the first conv accepts grayscale
        # without discarding the Kinetics-400 pretraining.
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

        # Remove the original 400-class Kinetics head. We rebuild the
        # classifier from scratch as two task-specific heads below.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.use_shared_head = use_shared_head
        if use_shared_head:
            # Shared MLP embedding feeds both task heads — our extension
            # beyond the Stanford paper's single-head design.
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
        else:
            # Ablation: skip the shared MLP, predict directly from the
            # 512-dim backbone features via two independent linear heads.
            self.shared_head = nn.Flatten()
            self.regression_head = nn.Linear(512, 1)
            self.classification_head = nn.Linear(512, num_classes)

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

        features = self.backbone(x)            # (B, 512, 1, 1, 1) after avgpool
        embedding = self.shared_head(features)  # (B, 128)
        ef_regression = self.regression_head(embedding).squeeze(-1)   # (B,)
        ef_classification = self.classification_head(embedding)        # (B, C)

        return ef_regression, ef_classification

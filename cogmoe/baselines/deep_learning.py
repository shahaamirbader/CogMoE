"""
Deep learning baselines (VGG-style and ResNet-style) for cognitive load classification.

Adapts VGG and ResNet architectures for feature-vector inputs (not images).
Input: pre-extracted feature vectors (89 dimensions = 33 ECG + 16 EEG +
30 Gaze + 10 EDA). Output: binary CL logits [B, 2].

Paper Tables 3-7 comparison baselines.

Usage:
    model = VGGBaseline(in_dim=89, num_classes=2, variant="feat")
    logits = model(x)  # x: [B, 89]
"""

import torch
import torch.nn as nn


class VGGBaseline(nn.Module):
    """
    VGG-style fully-connected network adapted for feature vectors.

    Two variants:
        - 'feat': Takes concatenated pre-extracted feature vector (89-dim).
        - 'raw': Takes a different input dimension (specified via in_dim) for
          raw signal representations or alternative feature sets.

    Architecture (feat variant):
        Linear(89, 256) -> ReLU -> Linear(256, 256) -> ReLU ->
        Linear(256, 128) -> ReLU -> Linear(128, 2)

    Paper Tables 3-7: VGG baseline for cognitive load classification.

    Args:
        in_dim: int - input feature dimension (default 89 for concatenated features).
        num_classes: int - number of output classes (default 2 for binary CL).
        variant: str - 'feat' for pre-extracted features, 'raw' for raw inputs.
    """

    def __init__(self, in_dim=89, num_classes=2, variant="feat"):
        super().__init__()
        self.variant = variant
        self.in_dim = in_dim
        self.num_classes = num_classes

        if variant == "feat":
            self.network = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
        elif variant == "raw":
            # Wider network for higher-dimensional raw inputs
            self.network = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
        else:
            raise ValueError(f"Unknown variant '{variant}'. Choose 'feat' or 'raw'.")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape [B, in_dim] - input feature vector.

        Returns:
            logits: Tensor of shape [B, num_classes] - unnormalized class scores.
        """
        return self.network(x)


class _ResidualBlock(nn.Module):
    """
    Single residual block for feature vectors.

    Architecture:
        Linear(dim, dim) -> ReLU -> Linear(dim, dim) + skip connection -> ReLU

    Args:
        dim: int - input and output dimension.
        dropout: float - dropout rate.
    """

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, dim].

        Returns:
            Tensor of shape [B, dim] - residual output.
        """
        residual = x
        out = self.block(x)
        out = self.norm(out + residual)
        out = self.act(out)
        return out


class ResNetBaseline(nn.Module):
    """
    ResNet-style network with residual blocks adapted for feature vectors.

    Two variants:
        - 'feat': Takes concatenated pre-extracted feature vector (89-dim).
        - 'raw': Takes a different input dimension for alternative inputs.

    Architecture (feat variant):
        Linear(89, 256) -> LayerNorm -> ReLU ->
        ResidualBlock(256) -> ResidualBlock(256) ->
        Linear(256, 2)

    Paper Tables 3-7: ResNet baseline for cognitive load classification.

    Args:
        in_dim: int - input feature dimension (default 89 for concatenated features).
        num_classes: int - number of output classes (default 2 for binary CL).
        variant: str - 'feat' for pre-extracted features, 'raw' for raw inputs.
        hidden_dim: int - hidden dimension for residual blocks (default 256).
        num_blocks: int - number of residual blocks (default 2).
        dropout: float - dropout rate (default 0.2).
    """

    def __init__(self, in_dim=89, num_classes=2, variant="feat",
                 hidden_dim=256, num_blocks=2, dropout=0.2):
        super().__init__()
        self.variant = variant
        self.in_dim = in_dim
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape [B, in_dim] - input feature vector.

        Returns:
            logits: Tensor of shape [B, num_classes] - unnormalized class scores.
        """
        h = self.input_proj(x)       # [B, hidden_dim]
        h = self.res_blocks(h)       # [B, hidden_dim]
        logits = self.classifier(h)  # [B, num_classes]
        return logits

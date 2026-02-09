"""
Classification head for cognitive load prediction.
Paper Section 3.2: "The resulting z_hat is passed through a classification head
to yield the final prediction y."
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Global average pooling over sequence -> Dropout -> FC -> logits.
    For binary cognitive load classification (low vs. high).

    Args:
        d_model: int - input dimension from transformer.
        num_classes: int - 2 for binary CL.
        dropout: float.
    """

    def __init__(self, d_model, num_classes=2, dropout=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            logits: [batch, num_classes]
        """
        # Pool over sequence dimension
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, d_model]
        x = self.dropout(x)
        return self.fc(x)

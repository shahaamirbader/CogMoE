"""
BIOT (Biosignal Transformer) baseline for cognitive load classification.

Simplified implementation of BIOT: a transformer encoder that operates on
concatenated physiological features. Adapted for pre-extracted feature vectors.
Reference: Azizi & BabaAli (2024).

Input: pre-extracted feature vectors (89 dimensions = 33 ECG + 16 EEG +
30 Gaze + 10 EDA). Output: binary CL logits [B, 2].

Paper Tables 3-7 comparison baseline.

Usage:
    model = BIOT(in_dim=89, d_model=256, nhead=4, num_layers=2, num_classes=2)
    logits = model(x)  # x: [B, 89]
"""

import math
import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the BIOT transformer.

    Since we operate on feature-level tokens (each feature as a token),
    this provides position information to distinguish feature indices.

    Args:
        d_model: int - embedding dimension.
        max_len: int - maximum number of feature tokens.
        dropout: float - dropout rate applied after adding positional encoding.
    """

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, seq_len, d_model].

        Returns:
            Tensor of shape [B, seq_len, d_model] with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BIOT(nn.Module):
    """
    Simplified BIOT: transformer encoder for concatenated physiological features.

    Architecture:
        1. Linear projection from input dimension to d_model
        2. Sinusoidal positional encoding
        3. TransformerEncoder (num_layers layers, nhead attention heads)
        4. Global average pooling over the sequence dimension
        5. Fully-connected classification head

    The input feature vector is treated as a single-token sequence (seq_len=1)
    and linearly projected, then processed by a standard transformer encoder.
    This follows the BIOT paradigm of applying transformer attention to
    biosignal representations.

    Reference: Azizi & BabaAli (2024) - "BIOT: Biosignal Transformer for
    cross-data learning in the wild."

    Paper Tables 3-7: BIOT baseline for cognitive load classification.

    Args:
        in_dim: int - input feature dimension (default 89).
        d_model: int - transformer model dimension (default 256).
        nhead: int - number of attention heads (default 4).
        num_layers: int - number of transformer encoder layers (default 2).
        num_classes: int - number of output classes (default 2).
        dropout: float - dropout rate (default 0.1).
        d_ff: int - feedforward dimension in transformer layers (default 512).
    """

    def __init__(self, in_dim=89, d_model=256, nhead=4, num_layers=2,
                 num_classes=2, dropout=0.1, d_ff=512):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Linear projection from input features to d_model embedding space
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_enc = _PositionalEncoding(d_model, max_len=200, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape [B, in_dim] or [B, seq_len, in_dim].
               If 2D, it is unsqueezed to [B, 1, in_dim] (single-token sequence).

        Returns:
            logits: Tensor of shape [B, num_classes] - unnormalized class scores.
        """
        # Handle both 2D (feature vector) and 3D (sequence) inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, in_dim]

        # 1. Project to d_model
        h = self.input_proj(x)  # [B, seq_len, d_model]

        # 2. Add positional encoding
        h = self.pos_enc(h)  # [B, seq_len, d_model]

        # 3. Transformer encoder
        h = self.transformer_encoder(h)  # [B, seq_len, d_model]

        # 4. Global average pooling over sequence dimension
        h = h.mean(dim=1)  # [B, d_model]

        # 5. Classification
        logits = self.classifier(h)  # [B, num_classes]

        return logits

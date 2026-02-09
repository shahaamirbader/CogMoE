"""
Three specialized experts for CogMoE's signal-quality-guided MoE.
Paper Section 3.2.1 and Appendix A.7.
"""

import torch
import torch.nn as nn
from cogmoe.models.attention import CrossModalAttentionForCRE


class HighFidelityExpert(nn.Module):
    """
    HFE: For clean signals (SNR > 15 dB).
    Lightweight FFN with two fully connected layers and ReLU activations.

    Paper A.7: "HFE employs a lightweight FFN with two fully connected layers
    and ReLU activations to balance efficiency and performance."

    Args:
        d_model: int - input/output dimension (256).
        d_ff: int - hidden dimension (512).
        dropout: float.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return self.ffn(x)


class NoiseResilientExpert(nn.Module):
    """
    NRE: For noisy signals.
    Extended FFN with residual connections and noise-aware normalization.

    Paper A.7: "The NRE handles noisy signals using residual connections and
    noise-aware normalization, where the residual paths capture fine-grained
    details and the normalization mitigates variability from motion artifacts."

    Args:
        d_model: int.
        d_ff: int.
        dropout: float.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Noise-aware normalization + residual connection
        return x + self.ffn(self.norm(x))


class ContextualRefinementExpert(nn.Module):
    """
    CRE: For masked or preliminarily recovered inputs.
    Cross-modal attention at the embedding level + contextual FFN.

    Paper A.7: "CRE refines interpolated values by leveraging cross-modal
    embeddings, enhancing the reliability of missing data imputation."

    Args:
        d_model: int.
        d_ff: int.
        nhead: int - attention heads for cross-modal attention.
        dropout: float.
    """

    def __init__(self, d_model, d_ff, nhead=4, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossModalAttentionForCRE(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Cross-modal attention (self-attention on the fused features)
        refined = self.cross_attn(x)
        # FFN with residual
        return refined + self.ffn(self.norm(refined))

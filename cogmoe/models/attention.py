"""
Cross-modal attention mechanisms for CogMoE.
Paper Section 3.2: Cross-attention layers capture inter-modal dependencies.
"""

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    Fuses modality embeddings via multi-head cross-attention across modalities
    at each timestep. Produces a single unified representation.

    Paper Section 3.2: "Z_m passes through cross-attention layers... capturing
    inter-modal dependencies."

    Args:
        d_model: int - embedding dimension.
        nhead: int - number of attention heads.
        dropout: float.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mod_embeds):
        """
        Args:
            mod_embeds: list of [batch, seq_len, d_model] (one per modality).

        Returns:
            fused: [batch, seq_len, d_model] - fused representation.
        """
        batch, seq_len, d_model = mod_embeds[0].shape
        M = len(mod_embeds)

        # Stack modalities: [M, batch*seq_len, d_model]
        stacked = torch.stack(mod_embeds, dim=2)  # [batch, seq_len, M, d_model]
        stacked = stacked.reshape(batch * seq_len, M, d_model).permute(1, 0, 2)
        # stacked: [M, batch*seq_len, d_model]

        fused, _ = self.attn(stacked, stacked, stacked)
        # Average across modalities
        fused = fused.mean(dim=0)  # [batch*seq_len, d_model]
        fused = fused.reshape(batch, seq_len, d_model)
        return self.norm(fused)


class CrossModalAttentionForCRE(nn.Module):
    """
    Cross-modal attention at the embedding level, used inside the Contextual
    Refinement Expert (CRE).

    Paper Section 3.2.1: "CRE focuses on masked or preliminarily recovered inputs,
    refining them at the embedding level via cross-modal attention."

    Args:
        d_model: int.
        nhead: int.
        dropout: float.
    """

    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None):
        """
        Args:
            x: [batch, seq_len, d_model] - input to refine.
            context: optional [batch, seq_len, d_model] - cross-modal context.
                     If None, uses self-attention.

        Returns:
            [batch, seq_len, d_model] - refined representation.
        """
        if context is None:
            context = x
        attn_out, _ = self.attn(x, context, context)
        return self.norm(x + attn_out)

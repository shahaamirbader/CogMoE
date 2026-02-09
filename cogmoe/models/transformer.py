"""
CogMoE Transformer: self-attention + MoE FFN (replacing standard FFN).
Paper Section 3.2 and Appendix A.6.
"""

import torch
import torch.nn as nn
from cogmoe.models.gating import MoEFeedForward


class CogMoELayer(nn.Module):
    """
    Single transformer layer: multi-head self-attention + MoE FFN.

    Paper A.6: "Like traditional transformers, it uses multi-head self-attention
    followed by a feed-forward network. However, instead of a standard FFN,
    it integrates a Mixture of Experts module."

    Args:
        d_model: int - model dimension.
        nhead: int - number of attention heads.
        d_ff: int - hidden dimension for expert FFNs.
        num_experts: int - K=3.
        q_dim: int - quality vector dimension.
        dropout: float.
    """

    def __init__(self, d_model, nhead, d_ff, num_experts, q_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_ffn = MoEFeedForward(d_model, d_ff, num_experts, q_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, q):
        """
        Args:
            x: [batch, seq_len, d_model].
            q: [batch, q_dim] - quality vector.

        Returns:
            x: [batch, seq_len, d_model].
            gate_weights: [batch, seq_len, num_experts].
            expert_outs: list of [batch, seq_len, d_model].
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # MoE FFN with residual
        moe_out, gate_weights, expert_outs = self.moe_ffn(x, q)
        x = self.norm2(x + self.dropout(moe_out))

        return x, gate_weights, expert_outs


class CogMoETransformer(nn.Module):
    """
    Stack of CogMoELayers with final layer normalization.

    Args:
        d_model, nhead, d_ff, num_layers, num_experts, q_dim, dropout.
    """

    def __init__(self, d_model, nhead, d_ff, num_layers, num_experts, q_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CogMoELayer(d_model, nhead, d_ff, num_experts, q_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, q):
        """
        Args:
            x: [batch, seq_len, d_model].
            q: [batch, q_dim] - quality vector.

        Returns:
            x: [batch, seq_len, d_model].
            gate_weights_list: list of [batch, seq_len, K] per layer.
            expert_outs_list: list of list of expert outputs per layer.
        """
        gate_weights_list = []
        expert_outs_list = []
        for layer in self.layers:
            x, gate_weights, expert_outs = layer(x, q)
            gate_weights_list.append(gate_weights)
            expert_outs_list.append(expert_outs)
        x = self.norm(x)
        return x, gate_weights_list, expert_outs_list

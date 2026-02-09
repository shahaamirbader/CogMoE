"""
Dynamic Pathway Gating (DPG) and MoE FeedForward for CogMoE.
Paper Section 3.2.1, Equations 5-7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cogmoe.models.experts import HighFidelityExpert, NoiseResilientExpert, ContextualRefinementExpert


class DynamicPathwayGating(nn.Module):
    """
    Routes fused features to specialized experts based on signal quality.
    Paper Eq.5: g_k(z, q) = softmax(W_{g,k} . [z; q])

    Args:
        d_model: int - dimension of fused feature z.
        q_dim: int - dimension of quality vector (num_modalities).
        num_experts: int - K=3.
    """

    def __init__(self, d_model, q_dim, num_experts=3):
        super().__init__()
        self.gate = nn.Linear(d_model + q_dim, num_experts)

    def forward(self, z, q):
        """
        Args:
            z: [batch, seq_len, d_model] - fused features.
            q: [batch, q_dim] - normalized quality vector.

        Returns:
            weights: [batch, seq_len, num_experts] - routing weights.
        """
        # Expand q to match seq_len dimension
        if q.dim() == 2:
            q = q.unsqueeze(1).expand(-1, z.size(1), -1)  # [batch, seq_len, q_dim]

        cat = torch.cat([z, q], dim=-1)  # [batch, seq_len, d_model + q_dim]
        logits = self.gate(cat)  # [batch, seq_len, num_experts]
        return F.softmax(logits, dim=-1)


class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts feedforward: DPG + 3 specialized experts.
    Replaces standard FFN in transformer layers.

    Paper Eq.6: z_hat = sum_k g_k(z, q) * f_k(z)

    Args:
        d_model: int.
        d_ff: int - hidden dimension for expert FFNs.
        num_experts: int - K=3.
        q_dim: int - quality vector dimension.
        dropout: float - expert dropout rate.
    """

    def __init__(self, d_model, d_ff, num_experts=3, q_dim=4, dropout=0.1):
        super().__init__()
        assert num_experts == 3, "CogMoE uses exactly 3 experts: HFE, NRE, CRE"

        self.experts = nn.ModuleList([
            HighFidelityExpert(d_model, d_ff, dropout),
            NoiseResilientExpert(d_model, d_ff, dropout),
            ContextualRefinementExpert(d_model, d_ff, nhead=4, dropout=dropout),
        ])
        self.dpg = DynamicPathwayGating(d_model, q_dim, num_experts)

    def forward(self, x, q):
        """
        Args:
            x: [batch, seq_len, d_model] - input features.
            q: [batch, q_dim] - quality vector.

        Returns:
            out: [batch, seq_len, d_model] - weighted expert output.
            gate_weights: [batch, seq_len, num_experts] - routing weights.
            expert_outs: list of [batch, seq_len, d_model] - per-expert outputs.
        """
        gate_weights = self.dpg(x, q)  # [batch, seq_len, num_experts]
        expert_outs = [expert(x) for expert in self.experts]

        # Weighted combination: Eq.6
        stacked = torch.stack(expert_outs, dim=-1)  # [batch, seq_len, d_model, K]
        out = (stacked * gate_weights.unsqueeze(2)).sum(dim=-1)  # [batch, seq_len, d_model]

        return out, gate_weights, expert_outs

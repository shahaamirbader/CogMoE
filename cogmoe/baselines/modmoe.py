"""
ModMoE: Modality-based routing MoE baseline for cognitive load classification.

Instead of routing by signal quality (DPG), assigns one expert per modality.
This baseline demonstrates that quality-guided routing outperforms naive
modality-based routing (Table 9 comparison in the paper).

Architecture:
    ModalityEncoders -> CrossModalFusion -> ModMoETransformer -> ClassificationHead

The key difference: each modality token is always routed to its designated expert,
rather than using DPG which routes based on signal quality scores.

With 4 modalities (ECG, EEG, Gaze, EDA) and 3 experts, the mapping is:
    Expert 0 <- ECG  (modality 0)
    Expert 1 <- EEG  (modality 1)
    Expert 2 <- Gaze (modality 2) and EDA (modality 3, shared round-robin)

Args:
    cfg: dict - same config format as CogMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cogmoe.models.encoders import build_encoders
from cogmoe.models.attention import CrossModalFusion
from cogmoe.models.classification_head import ClassificationHead
from cogmoe.models.experts import (
    HighFidelityExpert,
    NoiseResilientExpert,
    ContextualRefinementExpert,
)


class ModalityGating(nn.Module):
    """
    Fixed modality-based routing: assigns each modality to a designated expert.

    Unlike DPG which computes soft routing weights from [z; q], this module
    produces hard routing weights based solely on the modality identity of each
    token in the fused sequence.

    With M modalities and K experts (M >= K), modalities are mapped to experts
    round-robin: modality i -> expert (i % K). The gate output is a one-hot
    vector over experts, smoothed slightly for gradient flow.

    Args:
        num_modalities: int - M, number of modalities.
        num_experts: int - K, number of experts.
        label_smoothing: float - small epsilon to avoid exact zeros in gate
            weights, enabling gradient flow to all experts.
    """

    def __init__(self, num_modalities, num_experts=3, label_smoothing=0.05):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_experts = num_experts
        self.label_smoothing = label_smoothing

        # Pre-compute fixed routing table: [M, K]
        # modality i -> expert (i % K) gets weight (1 - eps), others eps/(K-1)
        routing = torch.full((num_modalities, num_experts),
                             label_smoothing / max(num_experts - 1, 1))
        for m in range(num_modalities):
            routing[m, m % num_experts] = 1.0 - label_smoothing
        self.register_buffer("routing_table", routing)

    def forward(self, z, q):
        """
        Compute fixed gate weights based on modality identity.

        After cross-modal fusion, the fused sequence has seq_len tokens.
        Since the fusion averages across modalities at each timestep,
        we approximate per-token modality assignment by distributing
        modalities evenly across the sequence and cycling.

        Args:
            z: [batch, seq_len, d_model] - fused features (used for shape only).
            q: [batch, q_dim] - quality vector (unused; accepted for interface
               compatibility with DPG).

        Returns:
            weights: [batch, seq_len, num_experts] - routing weights.
        """
        batch, seq_len, _ = z.shape

        # Assign each sequence position to a modality in round-robin fashion
        mod_indices = torch.arange(seq_len, device=z.device) % self.num_modalities
        # Look up the routing weights: [seq_len, num_experts]
        weights = self.routing_table[mod_indices]
        # Broadcast to batch: [batch, seq_len, num_experts]
        weights = weights.unsqueeze(0).expand(batch, -1, -1)

        return weights


class ModMoEFeedForward(nn.Module):
    """
    Modality-routed MoE feedforward: fixed routing + 3 specialized experts.

    Same experts as CogMoE (HFE, NRE, CRE), but gated by modality identity
    instead of DPG signal quality. This enables a controlled comparison of
    routing strategies.

    Paper Table 9: ModMoE baseline for routing ablation.

    Args:
        d_model: int - input/output dimension.
        d_ff: int - hidden dimension for expert FFNs.
        num_experts: int - K=3.
        num_modalities: int - M=4.
        dropout: float.
    """

    def __init__(self, d_model, d_ff, num_experts=3, num_modalities=4, dropout=0.1):
        super().__init__()
        assert num_experts == 3, "ModMoE uses the same 3 experts as CogMoE: HFE, NRE, CRE"

        self.experts = nn.ModuleList([
            HighFidelityExpert(d_model, d_ff, dropout),
            NoiseResilientExpert(d_model, d_ff, dropout),
            ContextualRefinementExpert(d_model, d_ff, nhead=4, dropout=dropout),
        ])
        self.gating = ModalityGating(num_modalities, num_experts)

    def forward(self, x, q):
        """
        Args:
            x: [batch, seq_len, d_model] - input features.
            q: [batch, q_dim] - quality vector (passed through for interface
               compatibility but not used for routing).

        Returns:
            out: [batch, seq_len, d_model] - weighted expert output.
            gate_weights: [batch, seq_len, num_experts] - routing weights.
            expert_outs: list of [batch, seq_len, d_model] - per-expert outputs.
        """
        gate_weights = self.gating(x, q)  # [batch, seq_len, num_experts]
        expert_outs = [expert(x) for expert in self.experts]

        # Weighted combination (same as CogMoE Eq.6)
        stacked = torch.stack(expert_outs, dim=-1)  # [batch, seq_len, d_model, K]
        out = (stacked * gate_weights.unsqueeze(2)).sum(dim=-1)  # [batch, seq_len, d_model]

        return out, gate_weights, expert_outs


class ModMoELayer(nn.Module):
    """
    Single transformer layer: multi-head self-attention + modality-routed MoE FFN.

    Same structure as CogMoELayer, but uses ModMoEFeedForward (fixed routing)
    instead of MoEFeedForward (DPG routing).

    Args:
        d_model: int - model dimension.
        nhead: int - number of attention heads.
        d_ff: int - hidden dimension for expert FFNs.
        num_experts: int - K=3.
        num_modalities: int - M=4.
        dropout: float.
    """

    def __init__(self, d_model, nhead, d_ff, num_experts, num_modalities, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.moe_ffn = ModMoEFeedForward(
            d_model, d_ff, num_experts, num_modalities, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, q):
        """
        Args:
            x: [batch, seq_len, d_model].
            q: [batch, q_dim] - quality vector (passed through, not used for routing).

        Returns:
            x: [batch, seq_len, d_model].
            gate_weights: [batch, seq_len, num_experts].
            expert_outs: list of [batch, seq_len, d_model].
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Modality-routed MoE FFN with residual
        moe_out, gate_weights, expert_outs = self.moe_ffn(x, q)
        x = self.norm2(x + self.dropout(moe_out))

        return x, gate_weights, expert_outs


class ModMoETransformer(nn.Module):
    """
    Stack of ModMoELayers with final layer normalization.

    Drop-in replacement for CogMoETransformer with modality-based routing.

    Args:
        d_model, nhead, d_ff, num_layers, num_experts, num_modalities, dropout.
    """

    def __init__(self, d_model, nhead, d_ff, num_layers, num_experts,
                 num_modalities, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ModMoELayer(d_model, nhead, d_ff, num_experts, num_modalities, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, q):
        """
        Args:
            x: [batch, seq_len, d_model].
            q: [batch, q_dim] - quality vector (passed through for interface compat).

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


class ModMoE(nn.Module):
    """
    Modality-based routing MoE model for cognitive load prediction.

    Identical to CogMoE except expert routing is determined by modality
    identity rather than DPG signal quality scores. This controlled
    comparison isolates the benefit of quality-aware routing.

    Paper Table 9: ModMoE baseline for routing strategy ablation.

    Accepts the same config dict as CogMoE for easy drop-in comparison.

    Args:
        cfg: dict - full configuration with 'data', 'model', 'loss' sections.
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        modalities = cfg["data"]["modalities"]

        d_model = model_cfg["d_model"]
        nhead = model_cfg["nhead"]
        d_ff = model_cfg["d_ff"]
        num_layers = model_cfg["num_layers"]
        num_experts = model_cfg["num_experts"]
        num_modalities = len(modalities)
        dropout = model_cfg.get("expert_dropout", 0.15)

        # 1. Modality-specific encoders (shared with CogMoE)
        self.encoders = build_encoders(cfg)

        # 2. Cross-modal fusion (shared with CogMoE)
        self.fusion = CrossModalFusion(d_model, nhead, dropout)

        # 3. Modality-routed MoE Transformer
        self.transformer = ModMoETransformer(
            d_model, nhead, d_ff, num_layers, num_experts, num_modalities, dropout
        )

        # 4. Classification head (shared with CogMoE)
        self.classifier = ClassificationHead(d_model, num_classes=2, dropout=dropout)

        self.modalities = modalities

    def forward(self, inputs, quality, mask=None):
        """
        Full forward pass.

        Args:
            inputs: dict {modality: Tensor [batch, seq_len, in_dim]}.
            quality: Tensor [batch, num_modalities] - quality scores (passed
                through for interface compatibility; not used for routing).
            mask: optional Tensor [batch, num_modalities] - 1.0 for present modalities.

        Returns:
            dict with keys:
                'logits': [batch, num_classes]
                'gate_weights_list': list of [batch, seq_len, K] per layer
        """
        # 1. Encode each modality
        mod_embeds = []
        for mod in self.modalities:
            if mod in inputs:
                emb = self.encoders[mod](inputs[mod])
                # Apply mask if provided
                if mask is not None:
                    mod_idx = self.modalities.index(mod)
                    emb = emb * mask[:, mod_idx].unsqueeze(1).unsqueeze(2)
                mod_embeds.append(emb)

        if not mod_embeds:
            raise ValueError("No modality inputs provided")

        # 2. Cross-modal fusion
        fused = self.fusion(mod_embeds)  # [batch, seq_len, d_model]

        # 3. Modality-routed MoE Transformer
        transformed, gate_weights_list, expert_outs_list = self.transformer(
            fused, quality
        )

        # 4. Classification
        logits = self.classifier(transformed)

        return {
            "logits": logits,
            "gate_weights_list": gate_weights_list,
        }

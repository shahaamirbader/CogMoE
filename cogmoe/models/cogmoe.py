"""
Top-level CogMoE model combining all Stage 2 components.
Encoders -> CrossModalFusion -> CogMoETransformer -> ClassificationHead.
"""

import torch
import torch.nn as nn
from cogmoe.models.encoders import build_encoders
from cogmoe.models.attention import CrossModalFusion
from cogmoe.models.transformer import CogMoETransformer
from cogmoe.models.classification_head import ClassificationHead


class CogMoE(nn.Module):
    """
    Complete CogMoE model for cognitive load prediction.

    Architecture:
        1. Modality-specific encoders project each modality to d_model embeddings
        2. Cross-modal fusion combines modality embeddings via attention
        3. CogMoE Transformer with quality-guided MoE processes fused features
        4. Classification head produces binary CL prediction

    Model size target: ~2.27M params, 19.9 MB, 12.5M FLOPs.

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
        q_dim = len(modalities)  # quality vector dimension = num modalities
        dropout = model_cfg.get("expert_dropout", 0.15)

        # 1. Modality-specific encoders
        self.encoders = build_encoders(cfg)

        # 2. Cross-modal fusion
        self.fusion = CrossModalFusion(d_model, nhead, dropout)

        # 3. MoE Transformer
        self.transformer = CogMoETransformer(
            d_model, nhead, d_ff, num_layers, num_experts, q_dim, dropout
        )

        # 4. Classification head
        self.classifier = ClassificationHead(d_model, num_classes=2, dropout=dropout)

        self.modalities = modalities

    def forward(self, inputs, quality, mask=None):
        """
        Full forward pass.

        Args:
            inputs: dict {modality: Tensor [batch, seq_len, in_dim]}.
            quality: Tensor [batch, num_modalities] - quality scores.
            mask: optional Tensor [batch, num_modalities] - 1.0 for present modalities.

        Returns:
            dict with keys:
                'logits': [batch, num_classes]
                'gate_weights_list': list of [batch, seq_len, K] per layer
                'expert_outs_list': list of expert output lists per layer
                'fused_features': [batch, seq_len, d_model]
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

        # 3. MoE Transformer
        transformed, gate_weights_list, expert_outs_list = self.transformer(fused, quality)

        # 4. Classification
        logits = self.classifier(transformed)

        return {
            "logits": logits,
            "gate_weights_list": gate_weights_list,
            "expert_outs_list": expert_outs_list,
            "fused_features": fused,
        }


def build_model(cfg):
    """Factory function to construct CogMoE from config."""
    return CogMoE(cfg)

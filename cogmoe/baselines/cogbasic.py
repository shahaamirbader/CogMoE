"""
CogBasic: Dense transformer baseline for cognitive load classification.

Same architecture as CogMoE but uses a standard FFN in the transformer instead
of MoE feed-forward. This baseline demonstrates the value of signal-quality-guided
expert routing (Table 8 comparison in the paper).

Architecture:
    ModalityEncoders -> CrossModalFusion -> TransformerEncoder (standard FFN) -> ClassificationHead

Args:
    cfg: dict - same config format as CogMoE.

Input: dict of modality tensors + quality vector.
Output: dict with 'logits' [B, 2].
"""

import torch
import torch.nn as nn
from cogmoe.models.encoders import build_encoders
from cogmoe.models.attention import CrossModalFusion
from cogmoe.models.classification_head import ClassificationHead


class DenseTransformerLayer(nn.Module):
    """
    Single transformer layer with standard FFN (no MoE).

    Same structure as CogMoELayer but replaces MoEFeedForward with a
    conventional two-layer FFN. This isolates the contribution of
    quality-guided expert routing.

    Architecture:
        MultiheadSelfAttention -> LayerNorm -> FFN -> LayerNorm

    Args:
        d_model: int - model dimension (256).
        nhead: int - number of attention heads.
        d_ff: int - hidden dimension for the feedforward network (512).
        dropout: float.
    """

    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model].

        Returns:
            x: [batch, seq_len, d_model].
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Standard FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class DenseTransformer(nn.Module):
    """
    Stack of DenseTransformerLayers with final layer normalization.

    Drop-in replacement for CogMoETransformer. Same d_model, nhead, d_ff,
    and num_layers, but without MoE routing.

    Args:
        d_model: int - model dimension.
        nhead: int - number of attention heads.
        d_ff: int - feedforward hidden dimension.
        num_layers: int - number of transformer layers.
        dropout: float.
    """

    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseTransformerLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model].

        Returns:
            x: [batch, seq_len, d_model].
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class CogBasic(nn.Module):
    """
    Dense transformer baseline: same backbone as CogMoE minus MoE routing.

    Shares the same modality-specific encoders, cross-modal fusion, and
    classification head as CogMoE. The only difference is the transformer
    body uses a standard feedforward network instead of the
    quality-guided Mixture-of-Experts (DPG + 3 experts).

    This allows direct ablation of the MoE contribution in Table 8.

    Model accepts the same config dict as CogMoE for easy comparison.

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
        dropout = model_cfg.get("expert_dropout", 0.15)

        # 1. Modality-specific encoders (shared with CogMoE)
        self.encoders = build_encoders(cfg)

        # 2. Cross-modal fusion (shared with CogMoE)
        self.fusion = CrossModalFusion(d_model, nhead, dropout)

        # 3. Dense Transformer (standard FFN, no MoE)
        self.transformer = DenseTransformer(d_model, nhead, d_ff, num_layers, dropout)

        # 4. Classification head (shared with CogMoE)
        self.classifier = ClassificationHead(d_model, num_classes=2, dropout=dropout)

        self.modalities = modalities

    def forward(self, inputs, quality, mask=None):
        """
        Full forward pass. Accepts quality for interface compatibility with
        CogMoE, but it is unused since there is no quality-guided routing.

        Args:
            inputs: dict {modality: Tensor [batch, seq_len, in_dim]}.
            quality: Tensor [batch, num_modalities] - quality scores (unused).
            mask: optional Tensor [batch, num_modalities] - 1.0 for present modalities.

        Returns:
            dict with keys:
                'logits': [batch, num_classes]
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

        # 3. Dense Transformer (quality vector is intentionally unused)
        transformed = self.transformer(fused)  # [batch, seq_len, d_model]

        # 4. Classification
        logits = self.classifier(transformed)

        return {
            "logits": logits,
        }

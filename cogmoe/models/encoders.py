"""
Modality-specific encoders for CogMoE.
Projects each modality's features into a shared d_model embedding space.
Paper Section 3.2 and Appendix A.5.2.
"""

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel attention block for ECG encoder.
    Paper A.5.2: "For ECG, we incorporate a channel attention block
    using a squeeze-and-excitation mechanism."

    Args:
        channels: int - number of channels (d_model).
        reduction: int - reduction ratio for bottleneck.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [batch, seq_len, channels]
        b, t, c = x.shape
        # Global avg pool across seq_len
        y = x.mean(dim=1)  # [batch, channels]
        y = self.excitation(y)  # [batch, channels]
        return x * y.unsqueeze(1)  # [batch, seq_len, channels]


class ModalityEncoder(nn.Module):
    """
    Projects raw modality features into shared embedding space.
    Pipeline: Linear -> LayerNorm -> ReLU -> Dropout.
    Optionally adds SE channel attention (for ECG).

    Paper A.5.2: "Our encoders adopt a unified yet adaptable pipeline...
    projecting each signal into a shared 256-dimensional embedding space."

    Args:
        in_dim: int - input feature dimension for this modality.
        d_model: int - output embedding dimension (256).
        dropout: float - modality-specific dropout rate.
        use_channel_attention: bool - True for ECG encoder.
    """

    def __init__(self, in_dim, d_model, dropout=0.1, use_channel_attention=False):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.se = SqueezeExcitation(d_model) if use_channel_attention else None

    def forward(self, x):
        # x: [batch, seq_len, in_dim] -> [batch, seq_len, d_model]
        y = self.proj(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        if self.se is not None:
            y = self.se(y)
        return y


# Feature dimensions per modality from the CL-Drive combined CSV
MODALITY_IN_DIMS = {
    "ECG": 33,
    "EEG": 16,
    "Gaze": 30,
    "EDA": 10,
}


def build_encoders(cfg):
    """
    Factory: build modality-specific encoders with correct dropout rates
    and channel attention flags from config.

    Args:
        cfg: dict - full config with 'model' and 'data' sections.

    Returns:
        nn.ModuleDict mapping modality names to ModalityEncoder instances.
    """
    d_model = cfg["model"]["d_model"]
    dropout_cfg = cfg["model"].get("dropout", {})
    use_ecg_se = cfg["model"].get("use_ecg_channel_attention", True)
    modalities = cfg["data"]["modalities"]

    encoders = {}
    for mod in modalities:
        in_dim = MODALITY_IN_DIMS.get(mod, 10)
        drop = dropout_cfg.get(mod.lower(), 0.1)
        use_se = use_ecg_se and (mod == "ECG")
        encoders[mod] = ModalityEncoder(in_dim, d_model, drop, use_se)

    return nn.ModuleDict(encoders)

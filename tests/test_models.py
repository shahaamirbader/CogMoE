"""
Unit tests for CogMoE model components.
Tests encoder shapes, expert outputs, gating, transformer, and full forward pass.
"""

import torch
import pytest


def _make_cfg():
    """Create a minimal test config."""
    return {
        "data": {
            "modalities": ["ECG", "EEG", "Gaze", "EDA"],
            "feature_groups": {"ECG": "_ECG", "EEG": "_EEG", "Gaze": "_Gaze", "EDA": "_EDA"},
        },
        "model": {
            "d_model": 64,
            "nhead": 4,
            "d_ff": 128,
            "num_layers": 2,
            "num_experts": 3,
            "dropout": {"ecg": 0.1, "eeg": 0.1, "eda": 0.1, "gaze": 0.1},
            "expert_dropout": 0.1,
            "use_ecg_channel_attention": True,
        },
    }


class TestSqueezeExcitation:
    def test_output_shape(self):
        from cogmoe.models.encoders import SqueezeExcitation
        se = SqueezeExcitation(channels=64, reduction=16)
        x = torch.randn(2, 5, 64)
        out = se(x)
        assert out.shape == (2, 5, 64)


class TestModalityEncoder:
    def test_output_shape(self):
        from cogmoe.models.encoders import ModalityEncoder
        enc = ModalityEncoder(in_dim=33, d_model=64, dropout=0.1, use_channel_attention=False)
        x = torch.randn(2, 1, 33)
        out = enc(x)
        assert out.shape == (2, 1, 64)

    def test_with_se(self):
        from cogmoe.models.encoders import ModalityEncoder
        enc = ModalityEncoder(in_dim=33, d_model=64, dropout=0.1, use_channel_attention=True)
        x = torch.randn(2, 1, 33)
        out = enc(x)
        assert out.shape == (2, 1, 64)

    def test_build_encoders(self):
        from cogmoe.models.encoders import build_encoders
        cfg = _make_cfg()
        encoders = build_encoders(cfg)
        assert set(encoders.keys()) == {"ECG", "EEG", "Gaze", "EDA"}
        # ECG encoder should have SE block
        assert encoders["ECG"].se is not None
        assert encoders["EEG"].se is None


class TestExperts:
    def test_hfe(self):
        from cogmoe.models.experts import HighFidelityExpert
        hfe = HighFidelityExpert(d_model=64, d_ff=128, dropout=0.1)
        x = torch.randn(2, 5, 64)
        out = hfe(x)
        assert out.shape == (2, 5, 64)

    def test_nre(self):
        from cogmoe.models.experts import NoiseResilientExpert
        nre = NoiseResilientExpert(d_model=64, d_ff=128, dropout=0.1)
        x = torch.randn(2, 5, 64)
        out = nre(x)
        assert out.shape == (2, 5, 64)

    def test_cre(self):
        from cogmoe.models.experts import ContextualRefinementExpert
        cre = ContextualRefinementExpert(d_model=64, d_ff=128, nhead=4, dropout=0.1)
        x = torch.randn(2, 5, 64)
        out = cre(x)
        assert out.shape == (2, 5, 64)


class TestDPG:
    def test_gating_output(self):
        from cogmoe.models.gating import DynamicPathwayGating
        dpg = DynamicPathwayGating(d_model=64, q_dim=4, num_experts=3)
        z = torch.randn(2, 5, 64)
        q = torch.randn(2, 4)
        weights = dpg(z, q)
        assert weights.shape == (2, 5, 3)
        # Weights should sum to 1 along expert dimension
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)

    def test_moe_feedforward(self):
        from cogmoe.models.gating import MoEFeedForward
        moe = MoEFeedForward(d_model=64, d_ff=128, num_experts=3, q_dim=4, dropout=0.1)
        x = torch.randn(2, 5, 64)
        q = torch.randn(2, 4)
        out, gw, eo = moe(x, q)
        assert out.shape == (2, 5, 64)
        assert gw.shape == (2, 5, 3)
        assert len(eo) == 3
        for e in eo:
            assert e.shape == (2, 5, 64)


class TestTransformer:
    def test_cogmoe_layer(self):
        from cogmoe.models.transformer import CogMoELayer
        layer = CogMoELayer(d_model=64, nhead=4, d_ff=128, num_experts=3, q_dim=4, dropout=0.1)
        x = torch.randn(2, 5, 64)
        q = torch.randn(2, 4)
        out, gw, eo = layer(x, q)
        assert out.shape == (2, 5, 64)

    def test_cogmoe_transformer(self):
        from cogmoe.models.transformer import CogMoETransformer
        model = CogMoETransformer(d_model=64, nhead=4, d_ff=128, num_layers=2,
                                   num_experts=3, q_dim=4, dropout=0.1)
        x = torch.randn(2, 5, 64)
        q = torch.randn(2, 4)
        out, gw_list, eo_list = model(x, q)
        assert out.shape == (2, 5, 64)
        assert len(gw_list) == 2
        assert len(eo_list) == 2


class TestClassificationHead:
    def test_output_shape(self):
        from cogmoe.models.classification_head import ClassificationHead
        head = ClassificationHead(d_model=64, num_classes=2, dropout=0.1)
        x = torch.randn(2, 5, 64)
        logits = head(x)
        assert logits.shape == (2, 2)


class TestFullCogMoE:
    def test_forward_pass(self):
        from cogmoe.models.cogmoe import CogMoE
        cfg = _make_cfg()
        model = CogMoE(cfg)

        inputs = {
            "ECG": torch.randn(2, 1, 33),
            "EEG": torch.randn(2, 1, 16),
            "Gaze": torch.randn(2, 1, 30),
            "EDA": torch.randn(2, 1, 10),
        }
        quality = torch.randn(2, 4)
        mask = torch.ones(2, 4)

        result = model(inputs, quality, mask)
        assert result["logits"].shape == (2, 2)
        assert len(result["gate_weights_list"]) == 2
        assert result["fused_features"].shape == (2, 1, 64)

    def test_parameter_count(self):
        from cogmoe.models.cogmoe import CogMoE
        from cogmoe.utils.reproducibility import count_parameters
        cfg = _make_cfg()
        model = CogMoE(cfg)
        params = count_parameters(model)
        assert params > 0
        # With small d_model=64, should be much less than 2.27M
        assert params < 500_000

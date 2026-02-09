"""
Integration tests for CogMoE pipeline.
Tests that the full training loop works end-to-end on synthetic data.
"""

import torch
import numpy as np
import pytest


class TestFullForwardBackward:
    def test_training_step(self):
        """Test that a full training step (forward + backward) works."""
        from cogmoe.models.cogmoe import CogMoE
        from cogmoe.losses.cortex_loss import CORTEXLoss

        cfg = {
            "data": {
                "modalities": ["ECG", "EEG", "Gaze", "EDA"],
                "feature_groups": {"ECG": "_ECG", "EEG": "_EEG", "Gaze": "_Gaze", "EDA": "_EDA"},
            },
            "model": {
                "d_model": 32, "nhead": 4, "d_ff": 64,
                "num_layers": 1, "num_experts": 3,
                "dropout": {"ecg": 0.1, "eeg": 0.1, "eda": 0.1, "gaze": 0.1},
                "expert_dropout": 0.1, "use_ecg_channel_attention": False,
            },
        }

        model = CogMoE(cfg)
        criterion = CORTEXLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic batch
        inputs = {
            "ECG": torch.randn(4, 1, 33),
            "EEG": torch.randn(4, 1, 16),
            "Gaze": torch.randn(4, 1, 30),
            "EDA": torch.randn(4, 1, 10),
        }
        quality = torch.randn(4, 4)
        mask = torch.ones(4, 4)
        targets = torch.tensor([0, 1, 0, 1])

        # Forward
        result = model(inputs, quality, mask)

        # Loss
        loss_dict = criterion(
            result["logits"], targets,
            result["gate_weights_list"],
            result["expert_outs_list"],
            result["fused_features"],
            epoch=0,
        )

        # Backward
        optimizer.zero_grad()
        loss_dict["total"].backward()
        optimizer.step()

        # Verify gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_loss_decreases(self):
        """Test that loss decreases over a few training steps."""
        from cogmoe.models.cogmoe import CogMoE
        from cogmoe.losses.cortex_loss import CORTEXLoss

        cfg = {
            "data": {
                "modalities": ["ECG", "EEG"],
                "feature_groups": {"ECG": "_ECG", "EEG": "_EEG"},
            },
            "model": {
                "d_model": 32, "nhead": 4, "d_ff": 64,
                "num_layers": 1, "num_experts": 3,
                "dropout": {"ecg": 0.1, "eeg": 0.1},
                "expert_dropout": 0.1, "use_ecg_channel_attention": False,
            },
        }

        model = CogMoE(cfg)
        criterion = CORTEXLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed synthetic batch (overfit to it)
        torch.manual_seed(42)
        inputs = {"ECG": torch.randn(8, 1, 33), "EEG": torch.randn(8, 1, 16)}
        quality = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

        losses = []
        for _ in range(20):
            result = model(inputs, quality)
            loss_dict = criterion(
                result["logits"], targets,
                result["gate_weights_list"],
                result["expert_outs_list"],
                result["fused_features"],
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()
            losses.append(loss_dict["total"].item())

        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestCrossValidation:
    def test_stratified_splits(self):
        from cogmoe.data.cross_validation import get_segment_stratified_splits
        labels = np.array([0]*100 + [1]*100)
        splits = get_segment_stratified_splits(labels, n_folds=5, seed=42)
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == 200
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0


class TestAugmentation:
    def test_compose(self):
        from cogmoe.data.augmentation import build_augmentation
        cfg = {
            "enabled": True,
            "gaussian_noise_std": 0.05,
            "channel_dropout_prob": 0.1,
            "temporal_shift_max": 5,
        }
        aug = build_augmentation(cfg)
        assert aug is not None

        sample = {
            "ECG": torch.randn(1, 33),
            "EEG": torch.randn(1, 16),
            "label": 0,
            "quality": torch.randn(4),
            "mask": torch.ones(4),
        }
        result = aug(sample)
        assert "ECG" in result
        assert result["ECG"].shape == (1, 33)

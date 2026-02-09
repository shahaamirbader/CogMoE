"""
Unit tests for CORTEX loss components.
"""

import torch
import pytest


class TestTaskLoss:
    def test_cross_entropy(self):
        from cogmoe.losses.components import task_loss
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = task_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0


class TestGateRegularization:
    def test_balanced_weights(self):
        from cogmoe.losses.components import gate_regularization
        # Perfectly balanced: should give ~0 penalty
        gw = torch.ones(4, 5, 3) / 3.0
        reg = gate_regularization([gw], num_experts=3)
        assert reg.item() < 1e-6

    def test_collapsed_weights(self):
        from cogmoe.losses.components import gate_regularization
        # All weight on expert 0: should give high penalty
        gw = torch.zeros(4, 5, 3)
        gw[:, :, 0] = 1.0
        reg = gate_regularization([gw], num_experts=3)
        assert reg.item() > 0.1


class TestNoiseSuppression:
    def test_identical_gives_zero(self):
        from cogmoe.losses.components import noise_suppression_loss
        ref = torch.randn(4, 5, 64)
        loss = noise_suppression_loss([ref], ref)
        assert loss.item() < 1e-6

    def test_different_gives_positive(self):
        from cogmoe.losses.components import noise_suppression_loss
        ref = torch.randn(4, 5, 64)
        noisy = ref + torch.randn_like(ref)
        loss = noise_suppression_loss([noisy], ref)
        assert loss.item() > 0


class TestCORTEXLoss:
    def test_full_loss(self):
        from cogmoe.losses.cortex_loss import CORTEXLoss
        criterion = CORTEXLoss(gamma=0.75, lambda_=0.6, beta_init=1.0,
                                beta_max=0.2, alpha_decay=0.05)

        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        gw_list = [torch.softmax(torch.randn(4, 5, 3), dim=-1)]
        eo_list = [[torch.randn(4, 5, 64) for _ in range(3)]]
        fused = torch.randn(4, 5, 64)

        result = criterion(logits, targets, gw_list, eo_list, fused, epoch=0)
        assert "total" in result
        assert "task" in result
        assert "noise" in result
        assert "refinement" in result
        assert "gate_reg" in result
        assert "beta" in result
        assert result["total"].requires_grad

    def test_beta_decay(self):
        from cogmoe.losses.cortex_loss import CORTEXLoss
        criterion = CORTEXLoss(beta_init=1.0, beta_max=0.2, alpha_decay=0.05)
        # At epoch 0: beta = min(0.2, 1.0 / 1.0) = 0.2
        assert criterion.get_beta(0) == 0.2
        # At epoch 100: beta = min(0.2, 1.0 / 6.0) ≈ 0.167
        assert criterion.get_beta(100) < 0.2
        # At epoch 1000: very small
        assert criterion.get_beta(1000) < 0.05

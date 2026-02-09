"""
CORTEX Loss: Cognitive Routing and Temporal Expertise Loss for CogMoE.
Paper Section 3.2.2, Equations 8-9.

L_CORTEX = L_task + gamma * L_noise + lambda * L_refinement + beta * R_gate

With adaptive decay for beta:
    beta = min(beta_max, beta_init / (1 + alpha * t))
"""

import torch
import torch.nn as nn
from cogmoe.losses.components import task_loss, noise_suppression_loss, refinement_loss, gate_regularization


class CORTEXLoss(nn.Module):
    """
    Full CORTEX Loss combining task, noise suppression, refinement, and
    gating regularization with adaptive weighting.

    Optimal coefficients (paper A.9):
        gamma=0.75, lambda=0.6, beta_init=1.0, alpha=0.05, beta_max=0.2

    Args:
        gamma: float - noise suppression weight.
        lambda_: float - refinement weight.
        beta_init: float - initial gate regularization weight.
        beta_max: float - maximum beta value after decay.
        alpha_decay: float - decay rate for beta schedule.
        num_experts: int - K (default 3).
    """

    def __init__(self, gamma=0.75, lambda_=0.6, beta_init=1.0,
                 beta_max=0.2, alpha_decay=0.05, num_experts=3):
        super().__init__()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.alpha_decay = alpha_decay
        self.num_experts = num_experts

    def get_beta(self, epoch):
        """
        Compute adaptive beta for current epoch.
        Paper Eq.9: beta = min(beta_max, beta_init / (1 + alpha * t))
        """
        return min(self.beta_max, self.beta_init / (1 + self.alpha_decay * epoch))

    def forward(self, logits, targets, gate_weights_list, expert_outs_list,
                fused_features, epoch=0):
        """
        Compute full CORTEX loss.

        Args:
            logits: [batch, num_classes] - classification logits.
            targets: [batch] - integer class labels.
            gate_weights_list: list of [batch, seq_len, K] per transformer layer.
            expert_outs_list: list of [HFE_out, NRE_out, CRE_out] per layer,
                each [batch, seq_len, d_model].
            fused_features: [batch, seq_len, d_model] - clean reference
                (pre-MoE fused features used as reference for NRE/CRE losses).
            epoch: int - current training epoch for adaptive beta.

        Returns:
            dict with keys:
                'total': total CORTEX loss (scalar).
                'task': L_task.
                'noise': L_noise.
                'refinement': L_refinement.
                'gate_reg': R_gate.
                'beta': current beta value.
        """
        # L_task: cross-entropy classification loss
        l_task = task_loss(logits, targets)

        # L_noise: NRE outputs vs clean references (expert index 1 = NRE)
        nre_outs = [layer_outs[1] for layer_outs in expert_outs_list]
        l_noise = noise_suppression_loss(nre_outs, fused_features.detach())

        # L_refinement: CRE outputs vs clean references (expert index 2 = CRE)
        cre_outs = [layer_outs[2] for layer_outs in expert_outs_list]
        l_refinement = refinement_loss(cre_outs, fused_features.detach())

        # R_gate: balanced expert utilization
        r_gate = gate_regularization(gate_weights_list, self.num_experts)

        # Adaptive beta
        beta = self.get_beta(epoch)

        # Total CORTEX loss (Eq.8)
        total = l_task + self.gamma * l_noise + self.lambda_ * l_refinement + beta * r_gate

        return {
            "total": total,
            "task": l_task.detach(),
            "noise": l_noise.detach(),
            "refinement": l_refinement.detach(),
            "gate_reg": r_gate.detach(),
            "beta": beta,
        }

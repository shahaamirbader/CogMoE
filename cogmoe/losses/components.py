"""
Individual loss components for the CORTEX Loss.
Paper Section 3.2.2 and Equations 7-8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def task_loss(logits, targets):
    """
    L_task: Standard cross-entropy loss on MoE outputs.
    Paper Eq.8 first term.

    Args:
        logits: [batch, num_classes] - model output logits.
        targets: [batch] - integer class labels (0 or 1).

    Returns:
        scalar Tensor.
    """
    return F.cross_entropy(logits, targets)


def noise_suppression_loss(nre_outputs, clean_references):
    """
    L_noise: Guides the NRE to produce denoised outputs matching clean references.
    Paper Eq.8 second term.

    Args:
        nre_outputs: list over layers of [batch, seq_len, d_model] - NRE expert outputs.
        clean_references: [batch, seq_len, d_model] - clean feature representations.

    Returns:
        scalar Tensor.
    """
    losses = []
    for nre_out in nre_outputs:
        losses.append(F.mse_loss(nre_out, clean_references))
    return torch.stack(losses).mean()


def refinement_loss(cre_outputs, reference_features):
    """
    L_refinement: Guides the CRE to improve fidelity of recovered modalities.
    Paper Eq.8 third term.

    Args:
        cre_outputs: list over layers of [batch, seq_len, d_model] - CRE expert outputs.
        reference_features: [batch, seq_len, d_model] - reference features.

    Returns:
        scalar Tensor.
    """
    losses = []
    for cre_out in cre_outputs:
        losses.append(F.mse_loss(cre_out, reference_features))
    return torch.stack(losses).mean()


def gate_regularization(gate_weights_list, num_experts=3):
    """
    R_gate: Balanced expert utilization penalty.
    Paper Eq.7: R_gate = sum_k ( (1/N) sum_i g_k(z_i, q_i) - 1/K )^2

    This smooth squared-error penalty discourages over-reliance on any single
    expert while retaining flexibility.

    Args:
        gate_weights_list: list over layers of [batch, seq_len, num_experts].
        num_experts: int - K (default 3).

    Returns:
        scalar Tensor.
    """
    reg = torch.tensor(0.0, device=gate_weights_list[0].device)
    target = 1.0 / num_experts

    for gw in gate_weights_list:
        # Average gate weight per expert across batch and seq_len
        avg_weights = gw.mean(dim=(0, 1))  # [num_experts]
        reg = reg + ((avg_weights - target) ** 2).sum()

    return reg / len(gate_weights_list)

"""
Evaluation metrics for CogMoE: accuracy, F1, expert utilization, statistical tests.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_accuracy(y_true, y_pred):
    """Classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_f1(y_true, y_pred, average="binary"):
    """F1 score."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute full metric suite: accuracy, F1, precision, recall, AUC-ROC.

    Args:
        y_true: array-like of true labels.
        y_pred: array-like of predicted labels.
        y_prob: optional array-like of predicted probabilities for positive class.

    Returns:
        dict with keys: accuracy, f1, precision, recall, auc_roc.
    """
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = 0.0
    return metrics


def compute_expert_utilization(gate_weights_list):
    """
    Compute percentage of samples routed to each expert.
    Paper target: ~35% HFE, 33% NRE, 32% CRE.

    Args:
        gate_weights_list: list of Tensors [batch, seq_len, num_experts] per layer,
            or a single Tensor.

    Returns:
        dict mapping expert index to utilization percentage, e.g.
        {"HFE": 35.4, "NRE": 33.1, "CRE": 31.5}
    """
    expert_names = ["HFE", "NRE", "CRE"]

    if isinstance(gate_weights_list, torch.Tensor):
        gate_weights_list = [gate_weights_list]

    # Average gate weights across all layers, batches, and timesteps
    all_weights = []
    for gw in gate_weights_list:
        if isinstance(gw, torch.Tensor):
            gw = gw.detach().cpu().numpy()
        all_weights.append(gw)

    stacked = np.concatenate([w.reshape(-1, w.shape[-1]) for w in all_weights], axis=0)
    # Find dominant expert per sample
    assignments = np.argmax(stacked, axis=-1)
    num_experts = stacked.shape[-1]

    utilization = {}
    for k in range(min(num_experts, len(expert_names))):
        pct = float(np.mean(assignments == k) * 100)
        utilization[expert_names[k]] = round(pct, 1)

    return utilization


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    """
    Nadeau & Bengio (1999) corrected resampled t-test for comparing
    two models on k-fold cross-validation.

    Args:
        scores_a: np.ndarray of per-fold scores for model A.
        scores_b: np.ndarray of per-fold scores for model B.
        n_train: int - number of training samples per fold.
        n_test: int - number of test samples per fold.

    Returns:
        (t_statistic, p_value) tuple.
    """
    from scipy import stats

    diffs = np.array(scores_a) - np.array(scores_b)
    k = len(diffs)
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)

    # Corrected variance
    corrected_var = (1.0 / k + n_test / n_train) * var_diff

    if corrected_var < 1e-12:
        return 0.0, 1.0

    t_stat = mean_diff / np.sqrt(corrected_var)
    p_val = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    return float(t_stat), float(p_val)

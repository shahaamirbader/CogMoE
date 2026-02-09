"""
Cross-validation split generation for CogMoE experiments.
Paper Section 4.1: 10-fold segment-wise stratified CV for CL-Drive.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold


def get_segment_stratified_splits(labels, n_folds=10, seed=42):
    """
    Generate segment-wise stratified k-fold splits for CL-Drive.
    Each participant's segments are evenly distributed across folds.

    Args:
        labels: np.ndarray of shape (N,) - binary labels.
        n_folds: int - number of folds (default 10).
        seed: int - random seed.

    Returns:
        list of (train_indices, test_indices) tuples.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        splits.append((train_idx, test_idx))
    return splits


def get_subject_wise_splits(subject_ids, n_folds=10, seed=42):
    """
    Generate subject-wise k-fold splits.
    Ensures no subject overlap between train and test sets.

    Args:
        subject_ids: np.ndarray of shape (N,) - participant IDs per sample.
        n_folds: int - number of folds (default 10).
        seed: int - random seed (not used by GroupKFold, included for API consistency).

    Returns:
        list of (train_indices, test_indices) tuples.
    """
    gkf = GroupKFold(n_splits=n_folds)
    dummy_X = np.zeros(len(subject_ids))
    dummy_y = np.zeros(len(subject_ids))
    splits = []
    for train_idx, test_idx in gkf.split(dummy_X, dummy_y, groups=subject_ids):
        splits.append((train_idx, test_idx))
    return splits

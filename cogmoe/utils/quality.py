"""
Signal quality scoring for Dynamic Pathway Gating (DPG).
Computes per-modality quality scores: SNR, missing proportion, autocorrelation.
Paper Eq.4: q_m = SNR_m * (1 - p_missing,m) * r_auto,m
"""

import numpy as np


def compute_snr(features):
    """
    Estimate SNR for a feature vector using a self-referencing strategy.
    Corrupts the signal with Gaussian noise and measures the ratio of
    signal variance to estimated noise variance.

    Args:
        features: 1D np.ndarray - feature vector for one modality.

    Returns:
        float - linear SNR ratio (not dB). Clipped to [0, 100].
    """
    features = np.asarray(features, dtype=float)
    features = features[np.isfinite(features)]
    if len(features) < 2:
        return 0.0

    signal_var = np.var(features)
    # Self-referencing: add small Gaussian noise, measure residual
    noise = np.random.randn(len(features)) * 0.01
    corrupted = features + noise
    noise_var = np.var(corrupted - features)

    if noise_var < 1e-12:
        return 100.0

    snr = signal_var / noise_var
    return float(np.clip(snr, 0, 100))


def compute_missing_proportion(features):
    """
    Compute (1 - p_missing): proportion of non-NaN and non-zero values.

    Args:
        features: 1D np.ndarray.

    Returns:
        float in [0, 1].
    """
    features = np.asarray(features, dtype=float)
    if len(features) == 0:
        return 0.0
    valid = np.isfinite(features) & (features != 0)
    return float(np.mean(valid))


def compute_autocorrelation(features, lag_window=10):
    """
    Compute average autocorrelation of a feature vector within L_m lag window.
    Higher autocorrelation indicates more temporal consistency.

    Args:
        features: 1D np.ndarray.
        lag_window: int - number of lags to average over.

    Returns:
        float in [0, 1].
    """
    features = np.asarray(features, dtype=float)
    features = features[np.isfinite(features)]
    n = len(features)
    if n < 2:
        return 0.0

    features = features - np.mean(features)
    var = np.var(features)
    if var < 1e-12:
        return 1.0

    max_lag = min(lag_window, n - 1)
    autocorrs = []
    for lag in range(1, max_lag + 1):
        c = np.mean(features[:n - lag] * features[lag:]) / var
        autocorrs.append(c)

    if not autocorrs:
        return 0.0
    return float(np.clip(np.mean(autocorrs), 0, 1))


def compute_quality_score(features, lag_window=10):
    """
    Compute per-modality quality score q_m.
    Paper Eq.4: q_m = SNR_m * (1 - p_missing,m) * r_auto,m

    Args:
        features: 1D np.ndarray - feature vector for one modality.
        lag_window: int - lag window for autocorrelation.

    Returns:
        float - quality score (non-negative).
    """
    snr = compute_snr(features)
    completeness = compute_missing_proportion(features)
    autocorr = compute_autocorrelation(features, lag_window)
    return snr * completeness * autocorr


def compute_quality_vector(modality_features, lag_windows=None):
    """
    Compute normalized quality vector for all modalities.

    Args:
        modality_features: dict {modality_name: 1D np.ndarray}.
        lag_windows: optional dict {modality_name: int lag_window}.

    Returns:
        np.ndarray of shape (num_modalities,) - normalized quality scores.
    """
    modalities = list(modality_features.keys())
    scores = []
    for mod in modalities:
        lag = 10
        if lag_windows and mod in lag_windows:
            lag = lag_windows[mod]
        scores.append(compute_quality_score(modality_features[mod], lag))

    scores = np.array(scores, dtype=float)
    total = scores.sum()
    if total > 0:
        scores = scores / total
    return scores
